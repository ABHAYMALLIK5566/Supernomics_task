# Multi-Hop Reasoning System

## Overview

The Multi-Hop Reasoning System is designed to handle complex and huge legal queries that require iterative reasoning, query decomposition, and comprehensive synthesis. This system automatically detects when a query is too complex for simple RAG processing and routes it through a sophisticated multi-step reasoning pipeline.

## Key Features

### 1. **Automatic Complexity Detection**
- Analyzes query structure, length, and legal terminology
- Detects multi-concept queries, conditional reasoning, and comparative analysis
- Routes queries to appropriate processing pipeline

### 2. **Query Decomposition**
- Breaks down complex queries into focused sub-queries
- Uses LLM-powered decomposition for optimal sub-query generation
- Maintains logical relationships between sub-queries

### 3. **Iterative Reasoning**
- Executes each sub-query independently
- Tracks reasoning steps and intermediate results
- Builds comprehensive understanding through multiple hops

### 4. **Intelligent Synthesis**
- Combines results from all reasoning steps
- Resolves conflicts and contradictions
- Produces coherent, comprehensive final answers

### 5. **Persistent Storage**
- Stores complete reasoning chains for future reference
- Enables session-based tracking and retrieval
- Provides analytics and statistics

## Architecture

```
Query Input
    ↓
Complexity Analysis
    ↓
Query Decomposition (if complex)
    ↓
Multi-Step Reasoning Execution
    ↓
Result Synthesis
    ↓
Final Answer + Reasoning Chain Storage
```

## API Usage

### Basic Multi-Hop Query

```python
import requests

# Complex query that will trigger multi-hop reasoning
query = {
    "query": "Compare the enforcement mechanisms in Article 41 and Article 42 of the UN Charter, and explain how they differ from the collective security provisions in Article 51, including the procedural requirements and limitations for each approach.",
    "enable_multi_hop_reasoning": True,
    "session_id": "user_session_123"
}

response = requests.post("http://localhost:8000/query", json=query)
result = response.json()

print(f"Chain ID: {result['chain_id']}")
print(f"Complexity Level: {result['complexity_level']}")
print(f"Final Answer: {result['final_answer']}")
print(f"Reasoning Steps: {len(result['reasoning_steps'])}")
```

### Force Multi-Hop Reasoning

```python
# Force multi-hop reasoning even for simple queries
query = {
    "query": "What is Article 41 of the UN Charter?",
    "force_multi_hop": True,
    "session_id": "user_session_123"
}

response = requests.post("http://localhost:8000/query", json=query)
```

### Retrieve Reasoning Chains

```python
# Get reasoning chains for a session
chains_request = {
    "session_id": "user_session_123",
    "limit": 10
}

response = requests.post("http://localhost:8000/reasoning-chains", json=chains_request)
chains = response.json()

for chain in chains:
    print(f"Chain {chain['chain_id']}: {chain['original_query']}")
    print(f"Steps: {len(chain['reasoning_steps'])}")
    print(f"Confidence: {chain['overall_confidence']:.2f}")
```

### Get Specific Reasoning Chain

```python
# Get a specific reasoning chain by ID
chain_id = "abc123-def456-ghi789"
response = requests.get(f"http://localhost:8000/reasoning-chains/{chain_id}")
chain = response.json()

print(f"Original Query: {chain['original_query']}")
print(f"Final Answer: {chain['final_answer']}")
for i, step in enumerate(chain['reasoning_steps']):
    print(f"Step {i+1}: {step['input_query']}")
    print(f"Result: {step['output_result'][:100]}...")
```

## Complexity Levels

### Simple
- Single concept queries
- Basic factual questions
- Direct article/section references

**Example**: "What does Article 41 of the UN Charter state?"

### Moderate
- Multi-concept queries with simple relationships
- Procedural questions
- Basic comparative analysis

**Example**: "How do Articles 41 and 42 of the UN Charter differ?"

### Complex
- Multi-aspect analysis
- Conditional reasoning
- Cross-document comparisons
- Causal chain analysis

**Example**: "Compare the enforcement mechanisms in Article 41 and Article 42 of the UN Charter, and explain how they differ from the collective security provisions in Article 51."

### Very Complex
- Multi-document analysis
- Complex conditional reasoning
- Procedural workflows
- Comprehensive legal analysis

**Example**: "Analyze the complete enforcement framework under the UN Charter, including Articles 41, 42, and 51, their procedural requirements, limitations, and how they interact in different conflict scenarios, considering both historical applications and contemporary interpretations."

## Reasoning Step Types

### 1. Query Decomposition
- Initial analysis and breakdown of complex queries
- Generation of focused sub-queries

### 2. Sub-Query Execution
- Individual processing of decomposed queries
- RAG-based information retrieval and synthesis

### 3. Information Synthesis
- Combining results from multiple steps
- Resolving conflicts and contradictions

### 4. Conflict Resolution
- Identifying and addressing conflicting information
- Providing balanced analysis

### 5. Final Synthesis
- Comprehensive integration of all reasoning steps
- Production of coherent final answer

## Configuration

### Environment Variables

```bash
# Enable multi-hop reasoning by default
ENABLE_MULTI_HOP_REASONING=true

# OpenAI configuration for reasoning
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

### Request Parameters

```python
{
    "query": "Your complex legal query",
    "enable_multi_hop_reasoning": True,  # Enable automatic detection
    "force_multi_hop": False,            # Force even for simple queries
    "session_id": "optional_session_id", # For tracking
    "top_k": 8,                          # Documents per step
    "similarity_threshold": 0.25         # Relevance threshold
}
```

## Performance Considerations

### Response Times
- Simple queries: < 2 seconds
- Complex queries: 5-15 seconds
- Very complex queries: 15-30 seconds

### Optimization Features
- Caching of reasoning chains
- Parallel sub-query execution
- Efficient database storage
- Session-based optimization

### Scaling
- Horizontal scaling with load balancers
- Database connection pooling
- Redis caching for frequent queries
- Async processing for better throughput

## Monitoring and Analytics

### Reasoning Statistics

```python
# Get reasoning statistics
response = requests.get("http://localhost:8000/reasoning-statistics?days=30")
stats = response.json()

print(f"Total chains: {stats['total_reasoning_chains']}")
print(f"Average execution time: {stats['average_execution_time']:.2f}s")
print(f"Average confidence: {stats['average_confidence']:.2f}")
print(f"Complexity distribution: {stats['complexity_distribution']}")
```

### Key Metrics
- Query complexity distribution
- Average execution times
- Confidence scores
- Success rates
- Most common query patterns

## Error Handling

### Common Error Scenarios
1. **Query too ambiguous**: System requests clarification
2. **Insufficient information**: Clear indication of limitations
3. **Processing timeout**: Graceful degradation to simpler processing
4. **Storage failures**: In-memory fallback with warnings

### Error Response Format

```python
{
    "chain_id": "error",
    "original_query": "Your query",
    "complexity_level": "error",
    "final_answer": "Error in multi-hop reasoning: [error details]",
    "reasoning_steps": [],
    "total_execution_time": 0.0,
    "overall_confidence": 0.0,
    "citations": [],
    "metadata": {"error": "error_details"}
}
```

## Best Practices

### 1. Query Formulation
- Be specific about legal provisions
- Include context when comparing multiple concepts
- Use clear, structured language

### 2. Session Management
- Use consistent session IDs for related queries
- Leverage session-based reasoning chain retrieval
- Monitor session performance

### 3. Performance Optimization
- Use appropriate similarity thresholds
- Monitor reasoning statistics
- Implement proper caching strategies

### 4. Error Handling
- Implement retry logic for transient failures
- Provide fallback to simpler processing
- Log and monitor error patterns

## Future Enhancements

### Planned Features
1. **Adaptive Reasoning**: Dynamic adjustment of reasoning depth
2. **Cross-Language Support**: Multi-lingual legal document processing
3. **Temporal Reasoning**: Time-based legal analysis
4. **Precedent Integration**: Case law reasoning chains
5. **Collaborative Reasoning**: Multi-user reasoning sessions

### Integration Opportunities
1. **Legal Citation Networks**: Graph-based citation analysis
2. **Compliance Checking**: Automated compliance verification
3. **Contract Analysis**: Multi-hop contract interpretation
4. **Regulatory Impact**: Cross-regulation analysis

## Troubleshooting

### Common Issues

1. **Slow Response Times**
   - Check database performance
   - Verify OpenAI API response times
   - Monitor system resources

2. **Low Confidence Scores**
   - Review document quality
   - Adjust similarity thresholds
   - Check query specificity

3. **Missing Reasoning Steps**
   - Verify query complexity detection
   - Check decomposition logic
   - Review error logs

### Debug Mode

Enable debug logging for detailed reasoning trace:

```python
import logging
logging.getLogger("app.services.multi_hop_reasoning").setLevel(logging.DEBUG)
```

This will provide detailed information about:
- Query complexity analysis
- Sub-query generation
- Step-by-step execution
- Synthesis process
- Storage operations
