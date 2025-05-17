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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from typing import List, Dict, Any, Optional, Tuple, Union

load_dotenv()

# Truth-Seeking MCP version number (increment with each change)
TRUTH_MCP_VERSION = "1.3.0"

# Default user ID for memory operations if using mem0
DEFAULT_USER_ID = "user"

# Create a dataclass for our application context
@dataclass
class TruthSeekingContext:
    """Context for the Truth-Seeking MCP server."""
    brave_search_api_key: str
    perplexity_api_key: Optional[str] = None
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