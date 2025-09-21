# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ECO1465 term project that analyzes social pressure and voting behavior using experimental data. The project replicates analysis from Gerber & Green's research on social pressure in voting and will be using causal ML methods to extend the analysis. The project is incomplete right now

## Commands

### Running Analysis
```bash
cd code
python main.py
```
This executes the full analysis pipeline, loading data and generating summary statistics. The intention is to call on functions from other modules and produce all outputs upon running main.py

### Running Individual Components
```bash
cd code
python analysis.py  # Generate summary statistics and LaTeX table
```

## Code Architecture

### Core Structure
- `code/main.py` - Entry point that orchestrates the analysis pipeline. The intention is to call on functions from other modules and produce all outputs upon running main.py
- `code/analysis.py` - Contains the analysis functions as needed to generate summary statistics and LaTeX tables.
- `code/utils.py` - Provides `get_project_paths()` utility for consistent path management across modules. Will contain future helper functions as needed.

### Data Flow
1. `main.py` loads `Data/social.csv` using pandas
2. Data processing includes:
   - Converting 'voted' from yes/no to binary (1/0)
   - Creating treatment dummy variables from categorical treatment column
   - Generating summary statistics for demographic and voting variables
3. Output is written to `Output/Tables/table1.tex` as formatted LaTeX

### Key Variables in Dataset
- Treatment groups: control, self, civic duty, neighbors, hawthorne
- Voting history: g2000, g2002, g2004 (general elections), p2000, p2002 (primaries)
- Demographics: sex, yob (year of birth)
- Outcome: voted (yes/no, converted to 1/0)

### Directory Structure
- `Data/` - Contains `social.csv` experimental dataset
- `Output/Tables/` - LaTeX table outputs (.tex files)
- `Output/Plots/` - Reserved for plots (currently empty)
- `Text/` - Contains reference PDF (`gerber_green.pdf`)
- `code/` - All Python analysis scripts

### Path Management
The `utils.get_project_paths()` function provides consistent paths:
- Uses `os.path` for cross-platform compatibility
- Returns dictionary with 'data', 'plots', 'tables', and 'parent_dir' keys
- All scripts should use this function rather than hardcoding paths