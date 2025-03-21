# TITAN Documentation Naming Conventions

This document establishes consistent naming conventions for all documentation in the TITAN Trading System project.

## File Naming Rules

1. **Use snake_case for all documentation files**
   - Example: `parameter_management.md` (correct)
   - Not: `parameter-management.md` or `ParameterManagement.md` (incorrect)

2. **All documentation files should use the `.md` extension**
   - We standardize on Markdown for all documentation

3. **Directory names should be lowercase with no separators**
   - Example: `docs/components/` (correct)
   - Not: `docs/Components/` or `docs/component-docs/` (incorrect)

4. **Use descriptive, specific names**
   - Example: `regime_detection_algorithm.md` (specific)
   - Not: `algorithm.md` (too vague)

5. **Avoid using dates in filenames**
   - Exception: For result reports that are explicitly time-based

## Directory Structure

```
/docs/
  ├── components/     # Component-specific documentation
  ├── developer/      # Documentation for developers
  ├── user/           # Documentation for end users
  ├── project/        # Project management documentation
  ├── testing/        # Testing-related documentation
  ├── results/        # Results and reports
  └── integration/    # Integration documentation
```

## Naming Pattern by Document Type

| Document Type | Naming Pattern | Example |
|---------------|----------------|---------|
| Architecture | `architecture_[aspect].md` | `architecture_overview.md` |
| Component | `[component_name].md` | `parameter_management.md` |
| Guide | `[audience]_guide_[topic].md` | `developer_guide_data_ingestion.md` |
| Reference | `[topic]_reference.md` | `api_reference.md` |
| Results | `results_[analysis_type].md` | `results_backtest_2025_03.md` |

## Document Headers

Each document should begin with a standardized header:

```markdown
# [Document Title]

**Type**: [Architecture|Component|Guide|Reference|Result]
**Last Updated**: YYYY-MM-DD
**Related Documents**: [Links to related docs]

[Brief description of document purpose]
```

## Converting Existing Documentation

When converting existing documentation to follow these conventions:

1. Create a new file with the correct naming convention
2. Copy and update the content
3. Update any cross-references
4. Update the documentation index
5. Archive the old document

## File Renaming Map

Below is a mapping of current documentation files to their new standardized names:

| Current Filename | New Standardized Filename |
|------------------|---------------------------|
| `architecture_document.md` | `architecture_overview.md` |
| `implementation-progress.md` | `project_implementation_status.md` |
| `development-tracker.md` | `project_development_tracker.md` |
| `TESTING_FRAMEWORK.md` | `testing_framework.md` |
| `ui_ux_architecture.md` | `architecture_ui_ux.md` |

## Exception Cases

Some exceptions to these naming conventions may be appropriate:

1. `README.md` files (keep as-is for platform compatibility)
2. Files with special significance (e.g., `LICENSE.md`, `CONTRIBUTING.md`)
3. Generated documentation from tools that enforce their own naming

## Implementation Timeline

1. Update documentation index immediately
2. Create template documents for missing sections
3. Rename high-priority documents within 1-2 weeks
4. Complete all documentation renaming within 1 month
5. Update all cross-references after renaming
