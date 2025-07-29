#!/bin/bash
# Humanoid Climb Project Activation Script

echo "🤖 Activating Humanoid Climb Virtual Environment..."

# Navigate to project directory
cd /Users/rwankhalifa/Documents/humanoid_climb

# Activate virtual environment
source venv/bin/activate

# Set default wandb entity for Minerva University
export WANDB_ENTITY="rawan_khalifa-minerva-university"

echo "✅ Virtual environment activated!"
echo "🏢 Wandb entity: $WANDB_ENTITY"
echo "📁 Current directory: $(pwd)"
echo "🐍 Python path: $(which python)"
echo ""
echo "🚀 Ready to train! Example commands:"
echo "   python train.py HumanoidClimb-v0 PPO -w 4 -t  # Uses Minerva org by default"
echo "   python test.py"
echo "   python climb.py"
echo "   python check_wandb.py  # Check wandb configuration"
echo ""
echo "To deactivate: deactivate"
