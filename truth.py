# ===============================================================
# Truth-Seeking MCP Server
# ===============================================================
#
# This MCP server offers the following tools:
#
# 1. analyze_claim - Analyzes a claim by finding supporting and opposing perspectives
#    from web searches, presenting balanced viewpoints with confidence scoring.
#
# 2. extract_claim - Extracts factual, predictive, or normative claims from text
#    using LLM processing, identifying substantial claims with confidence scores.
#
# 3. get_perspective_history - Retrieves historical perspectives for similar claims
#    from memory storage for consistent analysis over time.
#
# 4. transcribe_media - Transcribes speech from YouTube videos or podcasts into text
#    using OpenAI's Whisper API, supporting multiple languages.
#
# ===============================================================

from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
import httpx
import asyncio
import json
import os
import re
import logging
import datetime
import pathlib
import tempfile
import subprocess
import shutil

# For audio transcription
import yt_dlp
from pydub import AudioSegment
import openai

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from typing import List, Dict, Any, Optional, Tuple, Union

load_dotenv()

# Truth-Seeking MCP version number (increment with each change)
TRUTH_MCP_VERSION = "1.4.0"

# Default user ID for memory operations if using mem0
DEFAULT_USER_ID = "user"

# Create a dataclass for our application context
@dataclass
class TruthSeekingContext:
    """Context for the Truth-Seeking MCP server."""
    brave_search_api_key: str
    perplexity_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    mem0_client: Optional[Any] = None  # Will be initialized if memory integration is enabled

@asynccontextmanager
async def truth_seeking_lifespan(server: FastMCP) -> AsyncIterator[TruthSeekingContext]:
    """
    Manages the Truth-Seeking MCP server lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        TruthSeekingContext: The context containing API keys and clients
    """
    # Load API keys from environment
    brave_search_api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not brave_search_api_key:
        raise ValueError("BRAVE_SEARCH_API_KEY environment variable is required")
    
    # Perplexity API key is optional but recommended for confidence scoring
    
    # OpenAI API key for Whisper API
    openai_api_key = os.getenv("openai_api_key")
    if not openai_api_key:
        logging.warning("OpenAI API key not found, transcription functionality will be unavailable")
    
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    
    # Initialize mem0 client if enabled
    mem0_client = None
    if os.getenv("ENABLE_MEMORY", "false").lower() == "true":
        from utils import get_mem0_client
        mem0_client = get_mem0_client()
    
    try:
        yield TruthSeekingContext(
            brave_search_api_key=brave_search_api_key,
            perplexity_api_key=perplexity_api_key,
            openai_api_key=openai_api_key,
            mem0_client=mem0_client
        )
    finally:
        # No explicit cleanup needed for the clients
        pass

# Initialize FastMCP server
mcp = FastMCP(
    "truth-seeking-mcp",
    description="MCP server that finds supporting and opposing perspectives for claims",
    lifespan=truth_seeking_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8050")
)

@mcp.tool()
async def analyze_claim(ctx: Context, claim: str, claim_type: Optional[str] = None) -> str:
    """Analyze a claim by finding perspectives both supporting and opposing it.
    
    This tool searches the internet to find arguments on both sides of a claim,
    presenting balanced perspectives with sources.

    Args:
        ctx: The MCP server provided context
        claim: The claim to analyze (e.g., "Remote work improves productivity")
        claim_type: Optional type of claim - "factual", "predictive", or "normative"
                   If not provided, it will be automatically detected

    Returns:
        JSON formatted result with supporting and opposing perspectives
    """
    try:
        # Check if we have cached results if memory is enabled
        if ctx.request_context.lifespan_context.mem0_client:
            cached_result = await _check_memory_for_claim(ctx, claim)
            if cached_result:
                return cached_result
        
        # Auto-detect claim type if not provided
        if not claim_type:
            claim_type = await _detect_claim_type(claim)
        
        # Generate search queries for both perspectives
        supporting_query = await _generate_supporting_query(claim, claim_type)
        opposing_query = await _generate_opposing_query(claim, claim_type)
        
        # Execute searches sequentially with delay to avoid rate limiting
        supporting_search = await _search_brave(ctx, supporting_query)
        # Add a delay between API calls to avoid rate limiting
        await asyncio.sleep(1.2)  # Slightly more than 1 second to be safe
        opposing_search = await _search_brave(ctx, opposing_query)
        
        # Process and extract arguments
        supporting_args = await _extract_arguments(ctx, claim, supporting_search["results"], "supporting")
        opposing_args = await _extract_arguments(ctx, claim, opposing_search["results"], "opposing")
        
        # Capture total counts for prevalence calculation
        supporting_total = supporting_search["totalCount"]
        opposing_total = opposing_search["totalCount"]
        
        # Log the counts for debugging
        logging.info(f"Claim: '{claim[:50]}...', Supporting count: {supporting_total}, Opposing count: {opposing_total}")
        
        # Ensure we don't have zeros for both (which would result in 0.5 prevalence)
        if supporting_total == 0 and opposing_total == 0:
            # Use the number of arguments as a fallback
            supporting_total = len(supporting_args)
            opposing_total = len(opposing_args)
            logging.info(f"Using fallback counts - Supporting: {supporting_total}, Opposing: {opposing_total}")
        
        # Format results
        prevalence_score = _calculate_prevalence(supporting_total, opposing_total)
        result = {
            "version": TRUTH_MCP_VERSION,
            "claim": claim,
            "claim_type": claim_type,
            "timestamp": _get_current_timestamp(),
            "llm_confidence_analysis": any(arg.get("llm_analyzed", False) for arg in supporting_args + opposing_args),
            "perspectives": {
                "supporting": {
                    "arguments": supporting_args,
                    "total_results_found": supporting_total,
                    "prevalence_score": prevalence_score
                },
                "opposing": {
                    "arguments": opposing_args,
                    "total_results_found": opposing_total,
                    "prevalence_score": 1 - prevalence_score
                }
            }
        }
        
        # Save to memory if enabled
        if ctx.request_context.lifespan_context.mem0_client:
            await _save_result_to_memory(ctx, claim, result)
        
        # Save response to a local JSON file
        await _save_response_to_file(claim, result)
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error analyzing claim: {str(e)}"

async def _generate_supporting_query(claim: str, claim_type: str) -> str:
    """Generate a search query optimized for finding supporting arguments."""
    claim = claim.strip()
    
    if claim_type == "factual":
        return f"evidence supporting that {claim}"
    elif claim_type == "predictive":
        return f"predictions that {claim} analysis evidence"
    else:  # normative
        return f"arguments supporting why {claim} benefits advantages"

async def _generate_opposing_query(claim: str, claim_type: str) -> str:
    """Generate a search query optimized for finding opposing arguments."""
    claim = claim.strip()
    
    if claim_type == "factual":
        return f"evidence against {claim} disproven"
    elif claim_type == "predictive":
        return f"predictions against {claim} counterarguments evidence"
    else:  # normative
        return f"arguments against why {claim} disadvantages criticisms"

async def _search_brave(ctx: Context, query: str) -> Dict[str, Any]:
    """Execute a search using Brave Search API.
    
    Returns a dictionary containing the results and the total count of results found.
    """
    api_key = ctx.request_context.lifespan_context.brave_search_api_key
    max_results = 20  # Increased max results to get a better sample
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={"Accept": "application/json", "X-Subscription-Token": api_key}
        )
        
        if response.status_code == 200:
            response_json = response.json()
            # Log the full response for debugging - safely convert dict_keys to list
            try:
                logging.info(f"Brave Search API Response for query '{query[:30]}...': {list(response_json.keys())}")
            except Exception as e:
                logging.error(f"Error logging response keys: {str(e)}")
            
            results = response_json.get("web", {}).get("results", [])
            
            # Try different ways to get total count
            total_count = 0
            
            # Log structure for debugging - safely convert dict_keys to list
            if "web" in response_json:
                try:
                    logging.info(f"Web keys: {list(response_json['web'].keys())}")
                except Exception as e:
                    logging.error(f"Error logging web keys: {str(e)}")
                
            if "web" in response_json and "totalCount" in response_json["web"]:
                total_count = response_json["web"]["totalCount"]
            elif "approximateResultCount" in response_json:
                total_count = response_json["approximateResultCount"]
            elif "count" in response_json:
                total_count = response_json["count"]
            
            # Count results as a fallback
            if total_count == 0 and results:
                total_count = len(results) * 10  # Rough estimate: assume we're seeing ~10% of results
            
            logging.info(f"Query: '{query[:30]}...', Total count: {total_count}, Results count: {len(results)}")
            
            return {"results": results, "totalCount": total_count}
        elif response.status_code == 429:
            # Handle rate limiting more gracefully
            raise Exception(f"Rate limit exceeded. Please try again in a few seconds. Details: {response.text}")
        else:
            raise Exception(f"Search failed with status {response.status_code}: {response.text}")

async def _extract_arguments(ctx: Context, claim: str, search_results: List[Dict[str, Any]], perspective: str) -> List[Dict[str, Any]]:
    """Extract and format arguments from search results with confidence scoring."""
    arguments = []
    
    # Create base arguments with a default confidence score
    for result in search_results:
        text = result.get("description", "")
        if not text.strip():  # Skip empty descriptions
            continue
            
        arguments.append({
            "text": text,
            "source_url": result.get("url", ""),
            "source_title": result.get("title", ""),
            "confidence": 0.5,  # Default confidence, will be updated if LLM available
            "timestamp": _get_current_timestamp()
        })
    
    # Set default LLM usage flag
    llm_used_for_confidence = False
    
    # If Perplexity API key is available, enhance confidence scores with LLM
    if ctx.request_context.lifespan_context.perplexity_api_key and arguments:
        confidence_scores, llm_success = await _analyze_argument_confidence(ctx, claim, arguments, perspective)
        llm_used_for_confidence = llm_success
        
        # Update confidence scores
        for i, score in enumerate(confidence_scores):
            if i < len(arguments):  # Safety check
                arguments[i]["confidence"] = score
                
    # Add the LLM usage flag to each argument
    for arg in arguments:
        arg["llm_analyzed"] = llm_used_for_confidence
    
    return arguments

async def _analyze_argument_confidence(ctx: Context, claim: str, arguments: List[Dict[str, Any]], perspective: str) -> Tuple[List[float], bool]:
    """Use Perplexity API to analyze and score the confidence of arguments.
    
    Args:
        ctx: The MCP server context
        claim: The original claim being analyzed
        arguments: List of arguments to score
        perspective: Whether these are supporting or opposing arguments
        
    Returns:
        List of confidence scores (0.0 to 1.0) for each argument
    """
    api_key = ctx.request_context.lifespan_context.perplexity_api_key
    
    if not api_key or not arguments:
        return [0.7] * len(arguments), False  # Return default scores if no API key or arguments
        
    # Prepare the batch analysis prompt
    system_prompt = (
        "You are an objective evaluator of argument quality and relevance. "
        "Your task is to analyze arguments related to a specific claim and assign "
        "confidence scores from 0.0 to 1.0 based on:"
        "\n1. Relevance to the claim (is it directly addressing the claim?)"
        "\n2. Logical coherence (is the argument well-structured?)"
        "\n3. Factual basis (does it appear to be factual rather than opinioned?)"
        "\n4. Reliability of typical sources like this"
        "\n\nHigher scores indicate higher confidence that this is a strong and relevant argument."
    )
    
    argument_texts = [arg["text"] for arg in arguments[:10]]  # Limit to 10 to avoid token limits
    
    # Construct the user message for batch analysis
    user_message = f"Claim: {claim}\n\nAnalyze these {perspective} arguments and respond ONLY with a JSON array of confidence scores between 0.0 and 1.0:\n"
    
    for i, text in enumerate(argument_texts):
        user_message += f"\nArgument {i+1}: {text}"
    
    # Call Perplexity API to analyze the arguments
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": "llama-3-sonar-small-32k-online",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": 0.1,  # Low temperature for more consistent scoring
                    "max_tokens": 500
                },
                timeout=20.0  # Add timeout to prevent hanging
            )
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    # Try to extract the JSON array of scores
                    if "[" in content and "]" in content:
                        json_str = content[content.find("["):content.rfind("]")+1]
                        scores = json.loads(json_str)
                        
                        # Validate and normalize scores
                        valid_scores = []
                        for score in scores:
                            if isinstance(score, (int, float)):
                                # Ensure score is between 0.0 and 1.0
                                valid_scores.append(max(0.0, min(1.0, float(score))))
                            else:
                                valid_scores.append(0.7)  # Default for invalid scores
                                
                        # If we don't have enough scores, pad with defaults
                        while len(valid_scores) < len(arguments):
                            valid_scores.append(0.7)
                            
                        return valid_scores[:len(arguments)], True  # Return scores and success flag
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logging.error(f"Error processing Perplexity response content: {str(e)}")
    except Exception as e:
        logging.error(f"Error in confidence scoring: {str(e)}")
        
    # Fallback: return default scores if anything fails
    return [0.7] * len(arguments), False

async def _detect_claim_type(claim: str) -> str:
    """Automatically detect the type of claim based on its content."""
    claim = claim.lower()
    
    # Check for predictive language patterns
    predictive_patterns = ["will", "going to", "future", "predict", "forecast", "expect"]
    if any(pattern in claim for pattern in predictive_patterns):
        return "predictive"
    
    # Check for normative language patterns
    normative_patterns = ["should", "best", "better", "worse", "good", "bad", "optimal", "ideal", "right", "wrong"]
    if any(pattern in claim for pattern in normative_patterns):
        return "normative"
    
    # Default to factual if no other patterns match
    return "factual"

def _calculate_prevalence(supporting_count: int, opposing_count: int) -> float:
    """Calculate a prevalence score based on the total number of results found.
    
    Args:
        supporting_count: The total number of supporting results found (not just shown)
        opposing_count: The total number of opposing results found (not just shown)
        
    Returns:
        A score between 0 and 1 representing the prevalence of the supporting perspective
    """
    # Log the inputs
    logging.info(f"Calculating prevalence with: Supporting count = {supporting_count}, Opposing count = {opposing_count}")
    
    # Make sure we don't have negative counts
    supporting_count = max(0, supporting_count)
    opposing_count = max(0, opposing_count)
    
    # Ensure we have at least some data to work with
    if supporting_count == 0 and opposing_count == 0:
        logging.warning("Both supporting and opposing counts are 0, defaulting to 0.5 prevalence")
        return 0.5  # Equal if no results on either side
    
    total_count = supporting_count + opposing_count
    
    if total_count == 0:  # This should never happen given the above check
        return 0.5
    
    # Calculate base ratio
    raw_prevalence = supporting_count / total_count
    
    # Add a damping factor to avoid extreme scores when counts are very imbalanced
    # but still maintain the general trend
    damping = 0.1
    damped_prevalence = (1 - damping) * raw_prevalence + damping * 0.5
    
    logging.info(f"Raw prevalence: {raw_prevalence}, Damped prevalence: {damped_prevalence}")
    
    return damped_prevalence

def _get_current_timestamp() -> str:
    """Get the current timestamp in ISO format."""
    from datetime import datetime
    return datetime.utcnow().isoformat()

async def _check_memory_for_claim(ctx: Context, claim: str) -> Optional[str]:
    """Check if a similar claim has been analyzed before."""
    if not ctx.request_context.lifespan_context.mem0_client:
        return None
    
    mem0_client = ctx.request_context.lifespan_context.mem0_client
    search_results = mem0_client.search(claim, user_id=DEFAULT_USER_ID, limit=1)
    
    if isinstance(search_results, dict) and "results" in search_results:
        results = search_results["results"]
        if results and len(results) > 0:
            return json.dumps(results[0]["memory"], indent=2)
    
    return None

async def _save_result_to_memory(ctx: Context, claim: str, result: Dict[str, Any]) -> None:
    """Save the analysis result to memory."""
    if not ctx.request_context.lifespan_context.mem0_client:
        return
    
    mem0_client = ctx.request_context.lifespan_context.mem0_client
    memory_content = [{"role": "user", "content": json.dumps(result)}]
    mem0_client.add(memory_content, user_id=DEFAULT_USER_ID)

async def _save_response_to_file(claim: str, result: Dict[str, Any]) -> None:
    """Save the analysis result to a local JSON file.
    
    The filename will be formatted as: YYYY-MM-DD_HH-MM-SS_short-claim.json
    """
    try:
        # Create a short version of the claim for the filename (first 5 words)
        short_claim = "_".join(claim.split()[:5]).lower()
        # Remove special characters that aren't suitable for filenames
        short_claim = re.sub(r'[^\w\-\.]', '-', short_claim)
        
        # Format current datetime for filename
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create responses directory if it doesn't exist
        responses_dir = pathlib.Path("responses")
        responses_dir.mkdir(exist_ok=True)
        
        # Full filename with path
        filename = responses_dir / f"{date_str}_{short_claim}.json"
        
        # Write the JSON to file
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)
            
        logging.info(f"Saved response to file: {filename}")
    except Exception as e:
        logging.error(f"Error saving response to file: {str(e)}")
        # Don't raise the exception - this is a non-critical operation

@mcp.tool()
async def extract_claim(ctx: Context, text: str, max_claims: int = 5) -> str:
    """Extract factual, predictive, or normative claims from a large text input using LLM.
    
    Args:
        ctx: The MCP server provided context
        text: The text content to analyze for claims
        max_claims: Maximum number of claims to extract (default: 5)
        
    Returns:
        JSON formatted list of extracted claims with their types
    """
    try:
        # Set up the prompt for Perplexity API
        system_prompt = f"""
        You are a specialized claim extraction system designed to identify substantive claims from text.

        A substantive claim is a meaningful assertion that:
        1. Makes a factual statement about the world that can be verified or falsified
        2. Predicts future outcomes or trends
        3. Expresses a normative position about what should or ought to be done

        DO NOT identify as claims:
        - Simple statements of personal preference
        - Basic observations about immediate context ("It's Tuesday")
        - Greetings or conversational filler ("It is nice to see you")
        - Questions
        - Purely narrative or descriptive statements

        For each identified claim:
        1. Extract the exact text of the claim
        2. Classify it as:
           - "factual" (verifiable statements about reality)
           - "predictive" (forecasts about future events)
           - "normative" (value judgments or prescriptive statements)
        3. Provide a confidence score (0.1-1.0) indicating how clearly it represents a substantive claim

        Limit your response to the top {max_claims} most substantive claims.
        Response format: JSON array of claims, each with "claim", "claim_type", and "confidence" fields.
        If no substantive claims are found, return an empty array.
        """
        
        user_prompt = f"Extract substantive claims from the following text:\n\n{text}"
        
        # Make request to Perplexity API
        api_key = ctx.request_context.lifespan_context.perplexity_api_key
        if not api_key:
            logging.error("Perplexity API key not available for claim extraction")
            return json.dumps({
                "message": "Error: Perplexity API key not available for claim extraction",
                "claims": []
            }, indent=2)
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "sonar", 
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,  # Low temperature for more deterministic responses
            "max_tokens": 2048
        }
        
        logging.info("Making request to Perplexity API for claim extraction")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logging.error(f"Perplexity API error: {response.status_code} - {response.text}")
                return json.dumps({
                    "message": "Error extracting claims: API request failed",
                    "claims": []
                }, indent=2)
                
            response_data = response.json()
            llm_response = response_data["choices"][0]["message"]["content"]
            
            logging.info("Received response from Perplexity API")
            
            # Parse the response
            # The LLM should return JSON, but we'll handle cases where it doesn't
            try:
                # Find JSON in the response if the LLM added any extra text
                json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
                if json_match:
                    extracted_claims = json.loads(json_match.group(0))
                else:
                    extracted_claims = json.loads(llm_response)
                
                # If not a list (might be a dict with a 'claims' key), adjust accordingly
                if isinstance(extracted_claims, dict) and "claims" in extracted_claims:
                    extracted_claims = extracted_claims["claims"]
                
                # Ensure it's a list
                if not isinstance(extracted_claims, list):
                    extracted_claims = []
                    
                # Limit to max_claims
                extracted_claims = extracted_claims[:max_claims]
                
            except (json.JSONDecodeError, IndexError) as e:
                logging.error(f"Error parsing LLM response: {e}")
                logging.error(f"Raw response: {llm_response}")
                extracted_claims = []
        
        # Create result
        result = {
            "version": TRUTH_MCP_VERSION,
            "timestamp": _get_current_timestamp(),
            "message": f"Extracted {len(extracted_claims)} claim(s) from the provided text.",
            "claims": extracted_claims,
            "llm_extracted": True
        }
        
        # Save response to a local JSON file
        await _save_response_to_file("extracted_claims", result)
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error extracting claims: {str(e)}"

@mcp.tool()
async def get_perspective_history(ctx: Context, query: str, limit: int = 3) -> str:
    """Get historical perspectives for similar claims.
    
    Args:
        ctx: The MCP server provided context
        query: Search query to find similar historical claims
        limit: Maximum number of results to return
        
    Returns:
        JSON formatted list of historical claim analyses
    """
    if not ctx.request_context.lifespan_context.mem0_client:
        return "Memory integration is not enabled for this server"
    
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        memories = mem0_client.search(query, user_id=DEFAULT_USER_ID, limit=limit)
        
        if isinstance(memories, dict) and "results" in memories:
            flattened_memories = [memory["memory"] for memory in memories["results"]]
        else:
            flattened_memories = memories
            
        return json.dumps(flattened_memories, indent=2)
    except Exception as e:
        return f"Error retrieving historical perspectives: {str(e)}"

async def _download_media(url: str, output_dir: str) -> str:
    """Download audio from a YouTube video or podcast URL.
    
    Args:
        url: URL to YouTube video or podcast
        output_dir: Directory to save downloaded audio
        
    Returns:
        Path to downloaded audio file
    """
    logging.info(f"Downloading media from: {url}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            return filename
    except Exception as e:
        logging.error(f"Error downloading media: {str(e)}")
        raise

async def _process_audio_for_transcription(audio_path: str) -> str:
    """Process audio file for transcription, ensuring it meets Whisper API requirements.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Path to the processed audio file ready for transcription
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(audio_path)
        
        # Whisper has a 25MB limit, so we may need to reduce quality or split
        processed_path = audio_path.replace('.mp3', '_processed.mp3')
        
        # If file is too large, reduce quality
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # Size in MB
        
        if file_size > 24:  # Leave buffer under 25MB
            # Reduce bitrate to lower file size
            audio.export(processed_path, format="mp3", bitrate="64k")
        else:
            # Use original file
            processed_path = audio_path
        
        return processed_path
    except Exception as e:
        logging.error(f"Error processing audio: {str(e)}")
        raise

async def _transcribe_with_whisper(ctx: Context, audio_path: str, language: str = "en") -> dict:
    """Transcribe audio using OpenAI Whisper API.
    
    Args:
        ctx: The MCP server context
        audio_path: Path to the audio file
        language: Language code for transcription
        
    Returns:
        Dictionary containing the transcription and metadata
    """
    api_key = ctx.request_context.lifespan_context.openai_api_key
    if not api_key:
        raise ValueError("OpenAI API key not available for transcription")
    
    # Initialize OpenAI client with new API format
    client = openai.OpenAI(api_key=api_key)
    logging.info(f"Transcribing audio file: {audio_path}")
    
    try:
        with open(audio_path, "rb") as audio_file:
            # Use new client.audio.transcriptions.create format
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language if language else None
            )
        
        return response
    except Exception as e:
        logging.error(f"Error transcribing audio: {str(e)}")
        raise

@mcp.tool()
async def transcribe_media(ctx: Context, url: str, language: str = "en") -> str:
    """Transcribe speech from YouTube videos or podcasts into text.
    
    Args:
        ctx: The MCP server provided context
        url: URL to YouTube video or podcast episode
        language: Language code for transcription (default: 'en' for English)
        
    Returns:
        JSON formatted transcription with metadata
    """
    # Check for ffmpeg installation first
    try:
        # Try to run ffmpeg to check if it's installed
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        error_message = (
            "FFmpeg is not installed or not in PATH. This tool requires ffmpeg for audio processing.\n"
            "Please install ffmpeg:\n"
            "- On macOS: brew install ffmpeg\n"
            "- On Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "- On Windows: Download from https://ffmpeg.org/download.html\n"
            "\nAfter installing, restart the server and try again."
        )
        return json.dumps({
            "error": error_message,
            "transcription": ""
        }, indent=2)
    
    try:
        if not ctx.request_context.lifespan_context.openai_api_key:
            return json.dumps({
                "error": "OpenAI API key not available for transcription",
                "transcription": ""
            }, indent=2)
        
        # Create a temporary directory for downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Download the media from URL
            logging.info(f"Starting transcription process for: {url}")
            audio_path = await _download_media(url, temp_dir)
            
            # Step 2: Process the audio for transcription
            processed_audio = await _process_audio_for_transcription(audio_path)
            
            # Step 3: Transcribe the audio
            transcription_result = await _transcribe_with_whisper(ctx, processed_audio, language)
            
            # Extract title from filename
            filename = os.path.basename(audio_path)
            title = os.path.splitext(filename)[0]
            
            # Step 4: Save the transcription to a file
            timestamp = _get_current_timestamp()
            output_filename = f"{timestamp}_{title[:30].replace(' ', '_')}_transcription.json"
            output_path = os.path.join("responses", output_filename)
            
            # Prepare response - new OpenAI API returns an object with a text property
            result = {
                "url": url,
                "title": title,
                "language": language,
                "timestamp": timestamp,
                "transcription": transcription_result.text if hasattr(transcription_result, 'text') else str(transcription_result)
            }
            
            # Save to file
            os.makedirs("responses", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            
            logging.info(f"Transcription complete and saved to: {output_path}")
            
            return json.dumps(result, indent=2)
    
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error in transcription process: {error_message}")
        return json.dumps({
            "error": error_message,
            "transcription": ""
        }, indent=2)

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())