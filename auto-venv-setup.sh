#!/bin/bash
# Setup script to configure auto-activation of virtual environment

# Get absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Create the .bashrc addition
cat > "${PROJECT_DIR}/bashrc_addition.txt" << 'EOF'
# Auto-activate virtual environment when entering EYES-WhisperingNeMo directory
function auto_activate_venv() {
  if [[ "$PWD" == *"EYES-WhisperingNeMo"* ]]; then
    # Check if we're already in the right virtual environment
    if [[ "$VIRTUAL_ENV" != *"nemo-asr-venv"* ]]; then
      # Check if .autoenv exists in the current directory
      if [[ -f "$PWD/.autoenv" ]]; then
        source "$PWD/.autoenv"
      fi
    fi
  elif [[ "$VIRTUAL_ENV" == *"nemo-asr-venv"* ]]; then
    # Deactivate if we leave the project directory and the venv is active
    deactivate 2>/dev/null || true
    echo "Deactivated nemo-asr-venv virtual environment."
  fi
}

# Add the function to the PROMPT_COMMAND to execute before each prompt
if [[ ! "$PROMPT_COMMAND" == *"auto_activate_venv"* ]]; then
  PROMPT_COMMAND="auto_activate_venv;$PROMPT_COMMAND"
fi
EOF

# Determine the correct RC file to use
RC_FILE="$HOME/.bashrc"
if [[ -f "$HOME/.bash_profile" ]]; then
  RC_FILE="$HOME/.bash_profile"
elif [[ "$SHELL" == *"zsh"* ]] && [[ -f "$HOME/.zshrc" ]]; then
  RC_FILE="$HOME/.zshrc"
fi

# Check if the bashrc_addition is already in the RC file
if ! grep -q "Auto-activate virtual environment when entering EYES-WhisperingNeMo directory" "$RC_FILE"; then
  # Add to RC file
  cat "${PROJECT_DIR}/bashrc_addition.txt" >> "$RC_FILE"
  echo "Added auto-activation to $RC_FILE"
  echo "Please run 'source $RC_FILE' or restart your terminal for changes to take effect."
else
  echo "Auto-activation already set up in $RC_FILE"
fi

# Make the .autoenv file executable
chmod +x "${PROJECT_DIR}/.autoenv"

echo "Setup complete! The nemo-asr-venv will automatically activate when you enter this directory." 