#!/usr/bin/env python3
"""
Example script demonstrating two-stage workflow generation

This script shows how to use the improved prompt engineering and
two-stage generation (description optimization + workflow generation).
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.generators.prompt_builder import PromptBuilder
from evaluation.generators.description_optimizer import DescriptionOptimizer
from evaluation.generators.llm_workflow_generator import LLMWorkflowGenerator


def main():
    """Example of two-stage workflow generation"""
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return
    
    # Example user query
    user_query = """
    Create a workflow that receives webhook notifications when a new order is placed,
    validates the order data, sends a confirmation email to the customer,
    and logs the order to a database.
    """
    
    print("=" * 80)
    print("Two-Stage Workflow Generation Example")
    print("=" * 80)
    print()
    
    # Initialize components
    print("1. Initializing components...")
    
    # Use improved prompt
    prompt_path = "evaluation/config/workflow_generation_prompt_improved.txt"
    if not Path(prompt_path).exists():
        # Fallback to base prompt if improved doesn't exist
        prompt_path = "evaluation/config/workflow_generation_prompt.txt"
        print(f"   Using base prompt: {prompt_path}")
    else:
        print(f"   Using improved prompt: {prompt_path}")
    
    prompt_builder = PromptBuilder(prompt_path, use_improved=True)
    
    # Initialize description optimizer
    description_optimizer = DescriptionOptimizer(api_key=api_key, model="gpt-4o-mini")
    
    # Initialize workflow generator with description optimizer
    workflow_generator = LLMWorkflowGenerator(
        openai_api_key=api_key,
        prompt_builder=prompt_builder,
        model="gpt-4o",
        temperature=0.3,
        description_optimizer=description_optimizer
    )
    
    print("   ✓ Components initialized")
    print()
    
    # Stage 1: Optimize description
    print("2. Stage 1: Optimizing description...")
    optimization_result = description_optimizer.optimize_description(user_query, "example_001")
    
    if optimization_result.get('error'):
        print(f"   ✗ Optimization failed: {optimization_result['error']}")
        return
    
    optimized_description = optimization_result['optimized_description']
    print(f"   ✓ Description optimized")
    print(f"   Original length: {len(user_query)} chars")
    print(f"   Optimized length: {len(optimized_description)} chars")
    print(f"   Optimization cost: {optimization_result['usage']['total_tokens']} tokens")
    print()
    
    # Stage 2: Generate workflow
    print("3. Stage 2: Generating workflow...")
    workflow_result = workflow_generator.generate_workflow(
        description=user_query,  # Pass original query
        template_id="example_001",
        use_two_stage=True  # Enable two-stage generation
    )
    
    if workflow_result.get('error'):
        print(f"   ✗ Generation failed: {workflow_result['error']}")
        return
    
    print(f"   ✓ Workflow generated")
    print(f"   Workflow generation cost: {workflow_result['usage']['workflow_generation']['total_tokens']} tokens")
    
    if 'total_tokens' in workflow_result['usage']:
        print(f"   Total cost (both stages): {workflow_result['usage']['total_tokens']} tokens")
    print()
    
    # Display results
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print()
    
    print("Optimized Description:")
    print("-" * 80)
    print(optimized_description[:500] + "..." if len(optimized_description) > 500 else optimized_description)
    print()
    
    if workflow_result.get('llm_response'):
        workflow_plan = workflow_result['llm_response'].get('workflowPlan', {})
        print("Generated Workflow:")
        print("-" * 80)
        print(f"Name: {workflow_plan.get('name', 'N/A')}")
        print(f"Nodes: {len(workflow_plan.get('nodes', []))}")
        print(f"Connections: {len(workflow_plan.get('connections', []))}")
        
        if workflow_result['llm_response'].get('reasoning'):
            print()
            print("Reasoning (Chain-of-Thought):")
            reasoning = workflow_result['llm_response']['reasoning']
            for key, value in reasoning.items():
                print(f"  {key}: {value[:100]}..." if len(str(value)) > 100 else f"  {key}: {value}")
    
    print()
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
