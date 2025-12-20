# Contributing to MathCoRL

Thank you for your interest in contributing to MathCoRL. This document provides guidelines for contributing to the project.

## Code Style

### Python Standards
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use descriptive variable names

### Documentation
- Add docstrings to all public functions using Google style
- Update relevant documentation in `docs/` when adding features
- Include usage examples for new methods

## Testing Requirements

### Before Submitting
```bash
# Run existing examples to verify no breaking changes
python -m mint.cli solve --method fpp --question "Test problem"
python generate_candidates.py --dataset SVAMP --n-candidates 5
python train_policy.py --dataset SVAMP --epochs 1

# Verify imports work
python -c "from mint.core import FunctionPrototypePrompting"
python -c "from mint.icrl.policy_network import PolicyNetwork"
```

### Adding New Features
- Test with at least 2 datasets
- Verify compatibility with both OpenAI and Claude providers
- Include reproducibility with `--seed` flag
- Test edge cases (empty inputs, invalid data)

## Pull Request Process

1. **Fork and Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement Changes**
   - Make focused, atomic commits
   - Write clear commit messages following conventional commits format
   - Update documentation as needed

3. **Test Thoroughly**
   - Run manual tests as described above
   - Verify no new errors with `python -m mint.cli --help`

4. **Submit PR**
   - Provide clear description of changes
   - Reference related issues if applicable
   - Include example usage if adding new features

## Research Contributions

### New Prompting Methods
- Implement in new file: `mint/your_method.py`
- Inherit from base class or follow existing patterns
- Add integration to CLI: `mint/cli.py`
- Document method in `docs/usage.md`
- Test on at least 2 datasets

### New ICL Strategies
- Add to `mint/icrl/` directory
- Implement evaluation interface matching existing methods
- Test with `run_comparison.py`
- Document selection rules in `docs/policy-selection-rules.md`

### New LLM Providers
- Add provider configuration to `mint/config.py`
- Implement cost tracking in `mint/tracking.py`
- Test with all prompting methods
- Document setup in `docs/` directory

### New Datasets
- Add to `datasets/` directory with train/test splits
- Update dataset loader in `mint/utils.py`
- Add metadata to `configs/hyperparameters.yaml`
- Document in `docs/datasets.md`

## Code Review Standards

- **Functionality**: Code works as intended
- **Simplicity**: Clear, maintainable implementation
- **Documentation**: Adequate comments and docstrings
- **Compatibility**: Works with existing infrastructure
- **Testing**: Adequately tested before submission

## Questions?

Open an issue for:
- Feature proposals
- Bug reports
- Implementation questions
- Documentation improvements

We aim to respond within 48 hours.
