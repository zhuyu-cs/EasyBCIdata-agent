find . -type d -name __pycache__ -exec rm -rf {} +

rm -rf venv
rm -rf easybcidata_agent.egg-info
rm -rf .pytest_cache
rm -rf .ruff_cache
rm -rf easybci_web/node_modules
rm -rf easybci_cli/web_dist