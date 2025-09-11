#!/usr/bin/env python3
"""
Test script for multi-hop reasoning system.

Demonstrates the capabilities of the multi-hop reasoning system with various
complexity levels and query types.
"""

import asyncio
import json
import time
from typing import Dict, Any

# Import the multi-hop reasoning components
from app.services.multi_hop_reasoning import multi_hop_reasoning_engine
from app.services.query_complexity_detector import query_complexity_detector
from app.services.reasoning_chain_storage import reasoning_chain_storage


class MultiHopReasoningTester:
    """Test class for multi-hop reasoning system"""
    
    def __init__(self):
        self.test_queries = [
            # Simple queries
            {
                "query": "What is Article 41 of the UN Charter?",
                "expected_complexity": "simple",
                "description": "Simple factual query"
            },
            
            # Moderate complexity
            {
                "query": "How do Articles 41 and 42 of the UN Charter differ in their enforcement mechanisms?",
                "expected_complexity": "moderate",
                "description": "Comparative analysis query"
            },
            
            # Complex queries
            {
                "query": "Compare the enforcement mechanisms in Article 41 and Article 42 of the UN Charter, and explain how they differ from the collective security provisions in Article 51, including the procedural requirements and limitations for each approach.",
                "expected_complexity": "complex",
                "description": "Multi-aspect comparative analysis"
            },
            
            # Very complex queries
            {
                "query": "Analyze the complete enforcement framework under the UN Charter, including Articles 41, 42, and 51, their procedural requirements, limitations, and how they interact in different conflict scenarios, considering both historical applications and contemporary interpretations of these provisions.",
                "expected_complexity": "very_complex",
                "description": "Comprehensive legal framework analysis"
            },
            
            # Conditional reasoning
            {
                "query": "If a state violates international law, what are the available enforcement mechanisms under the UN Charter, and how do the procedures differ depending on whether the violation constitutes a threat to international peace and security?",
                "expected_complexity": "complex",
                "description": "Conditional reasoning with procedural analysis"
            },
            
            # Multi-document analysis
            {
                "query": "Examine how the enforcement provisions in the UN Charter interact with the principles of sovereignty and non-intervention, and analyze the balance between collective security and state autonomy across different articles and chapters.",
                "expected_complexity": "very_complex",
                "description": "Cross-document principle analysis"
            }
        ]
    
    async def initialize_services(self):
        """Initialize all required services"""
        print("🔧 Initializing services...")
        
        try:
            # Initialize reasoning chain storage
            await reasoning_chain_storage.initialize()
            print("✅ Reasoning chain storage initialized")
            
            print("✅ All services initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Service initialization failed: {e}")
            return False
    
    async def test_complexity_detection(self):
        """Test query complexity detection"""
        print("\n🔍 Testing Query Complexity Detection")
        print("=" * 50)
        
        for i, test_case in enumerate(self.test_queries, 1):
            query = test_case["query"]
            expected = test_case["expected_complexity"]
            description = test_case["description"]
            
            print(f"\nTest {i}: {description}")
            print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            try:
                # Analyze complexity
                complexity, query_type, analysis = query_complexity_detector.detect_complexity_and_type(query)
                
                print(f"Detected Complexity: {complexity.value}")
                print(f"Query Type: {query_type.value}")
                print(f"Expected: {expected}")
                print(f"Match: {'✅' if complexity.value == expected else '❌'}")
                print(f"Complexity Score: {analysis.get('enhanced_complexity_score', 0):.1f}")
                print(f"Requires Multi-Hop: {'Yes' if analysis.get('requires_multi_hop', False) else 'No'}")
                
                if analysis.get('detected_indicators'):
                    print(f"Detected Patterns: {list(analysis['detected_indicators'].keys())}")
                
            except Exception as e:
                print(f"❌ Error in complexity detection: {e}")
    
    async def test_multi_hop_reasoning(self, max_queries: int = 2):
        """Test multi-hop reasoning with complex queries"""
        print(f"\n🧠 Testing Multi-Hop Reasoning (max {max_queries} queries)")
        print("=" * 50)
        
        # Test only complex and very complex queries
        complex_queries = [q for q in self.test_queries 
                          if q["expected_complexity"] in ["complex", "very_complex"]]
        
        for i, test_case in enumerate(complex_queries[:max_queries], 1):
            query = test_case["query"]
            description = test_case["description"]
            
            print(f"\n🧠 Multi-Hop Test {i}: {description}")
            print(f"Query: {query[:150]}{'...' if len(query) > 150 else ''}")
            
            try:
                start_time = time.time()
                
                # Process with multi-hop reasoning
                reasoning_chain = await multi_hop_reasoning_engine.process_complex_query(
                    query=query,
                    session_id=f"test_session_{i}"
                )
                
                execution_time = time.time() - start_time
                
                print(f"✅ Processing completed in {execution_time:.2f}s")
                print(f"Chain ID: {reasoning_chain.chain_id}")
                print(f"Complexity Level: {reasoning_chain.complexity_level.value}")
                print(f"Number of Steps: {len(reasoning_chain.steps)}")
                print(f"Overall Confidence: {reasoning_chain.overall_confidence:.2f}")
                print(f"Total Citations: {len(reasoning_chain.citations)}")
                
                # Display reasoning steps
                print("\n📋 Reasoning Steps:")
                for j, step in enumerate(reasoning_chain.steps, 1):
                    print(f"  Step {j} ({step.step_type.value}):")
                    print(f"    Input: {step.input_query[:80]}{'...' if len(step.input_query) > 80 else ''}")
                    print(f"    Confidence: {step.confidence_score:.2f}")
                    print(f"    Time: {step.execution_time:.2f}s")
                
                # Display final answer preview
                print(f"\n📝 Final Answer Preview:")
                answer_preview = reasoning_chain.final_answer[:200]
                print(f"  {answer_preview}{'...' if len(reasoning_chain.final_answer) > 200 else ''}")
                
                # Store the reasoning chain
                await reasoning_chain_storage.store_reasoning_chain(reasoning_chain)
                print(f"💾 Reasoning chain stored successfully")
                
            except Exception as e:
                print(f"❌ Multi-hop reasoning failed: {e}")
    
    async def test_reasoning_chain_retrieval(self):
        """Test reasoning chain storage and retrieval"""
        print(f"\n💾 Testing Reasoning Chain Storage & Retrieval")
        print("=" * 50)
        
        try:
            # Get reasoning statistics
            stats = await reasoning_chain_storage.get_reasoning_statistics(days=1)
            
            print("📊 Reasoning Statistics:")
            print(f"  Total Chains: {stats.get('total_reasoning_chains', 0)}")
            print(f"  Average Execution Time: {stats.get('average_execution_time', 0):.2f}s")
            print(f"  Average Confidence: {stats.get('average_confidence', 0):.2f}")
            
            complexity_dist = stats.get('complexity_distribution', {})
            print(f"  Complexity Distribution:")
            for level, count in complexity_dist.items():
                print(f"    {level}: {count}")
            
            # Test session-based retrieval
            test_session = "test_session_1"
            session_chains = await reasoning_chain_storage.get_reasoning_chains_by_session(test_session)
            
            print(f"\n🔍 Session Chains for '{test_session}': {len(session_chains)}")
            for chain in session_chains:
                print(f"  Chain {chain.chain_id}: {chain.original_query[:50]}...")
                print(f"    Steps: {len(chain.steps)}, Confidence: {chain.overall_confidence:.2f}")
            
        except Exception as e:
            print(f"❌ Reasoning chain retrieval failed: {e}")
    
    async def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        print("🚀 Multi-Hop Reasoning System Test Suite")
        print("=" * 60)
        
        # Initialize services
        if not await self.initialize_services():
            return
        
        # Test complexity detection
        await self.test_complexity_detection()
        
        # Test multi-hop reasoning (limited to avoid long execution)
        await self.test_multi_hop_reasoning(max_queries=2)
        
        # Test storage and retrieval
        await self.test_reasoning_chain_retrieval()
        
        print(f"\n✅ Test suite completed successfully!")
        print("=" * 60)


async def main():
    """Main test function"""
    tester = MultiHopReasoningTester()
    await tester.run_comprehensive_test()


if __name__ == "__main__":
    print("Multi-Hop Reasoning System Test")
    print("Note: This test requires the application to be running with proper database setup")
    print("Make sure to set OPENAI_API_KEY environment variable")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
