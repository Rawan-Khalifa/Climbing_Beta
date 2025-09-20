#!/bin/bash

echo "🧹 Cleaning up project structure..."

# Remove trial/experimental files
echo "Removing trial files..."
rm -f advanced_video_analysis.py
rm -f climbing_analysis.py
rm -f behavior_extraction.py
rm -f trajectory_to_actions.py
rm -f bc_rl_hybrid.py
rm -f complete_bc_rl_pipeline.py
rm -f enhanced_ppo.py
rm -f smart_video_rl.py
rm -f train_with_videos.py
rm -f check_wandb.py
rm -f behavioral_cloning.py
rm -f diagnostic_test.py
rm -f test_curriculum.py
rm -f video_analysis_pipeline.py
rm -f video_to_demo.py

# Remove old analysis directories
echo "Removing old analysis directories..."
rm -rf advanced_video_analysis/
rm -rf trajectories/
rm -rf warped_videos/
rm -rf analysis/
rm -rf video_analysis_output/
rm -rf logs/
rm -rf models/

# Remove the climbing-analysis-toolbox (keeping only what we need)
echo "Removing external toolbox..."
rm -rf climbing-analysis-toolbox/

# Remove old demo and state files that aren't needed
echo "Cleaning demonstrations directory..."
rm -f demonstrations/bc_training_demos.pkl

# Remove wandb logs (keep if you want training history)
echo "Cleaning wandb logs..."
# rm -rf wandb/  # Uncomment if you want to remove training history

# Remove Python cache files
echo "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# Remove other temporary files
echo "Removing temporary files..."
rm -f reference_wall.jpg
rm -f current_frame.jpg
rm -f *.log

# Create .gitignore for clean repository
echo "Creating .gitignore..."
cat > .gitignore << EOF
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so

# Virtual environments
venv/
env/
.env

# Training outputs
extracted_poses/
demonstrations/
best_bc_guided_model/
ppo_bc_logs/
eval_logs/
*.pkl
*.pth

# Videos (add your own)
videos/*.mp4
climbing_videos/*.mp4

# Logs
*.log
wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
tmp/
temp/
*.tmp
EOF

echo "✅ Cleanup complete!"
echo ""
echo "📁 Clean project structure:"
echo "├── core/                    # Core pipeline"
echo "├── humanoid_climb/          # Environment & assets"
echo "├── videos/                  # Your climbing videos"
echo "├── run_pipeline.py          # Main pipeline runner"
echo "├── requirements_clean.txt   # Dependencies"
echo "└── README_CLEAN.md          # Documentation"
echo ""
echo "🚀 Ready to use! Run: python run_pipeline.py --videos-dir videos"
