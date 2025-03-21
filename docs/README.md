# TITAN Trading System Documentation

This directory contains the documentation for the TITAN Trading System.

## Documentation Structure

The documentation is organized into the following categories:

- **Architecture**: System-level architecture and design documents
- **Components**: Documentation for specific components of the system
- **Developer**: Guides and references for developers
- **User**: Guides for end users
- **Project**: Project management documentation
- **Testing**: Testing framework and procedures
- **Results**: Analysis and reports from system runs
- **Integration**: Documentation for integration with external systems
- **Archive**: Old documentation files kept for historical reference

## Main Index

The main entry point for all documentation is [index.md](./index.md), which provides a structured overview of all available documentation.

## Documentation Standards

The following standards apply to all documentation:

1. **File Naming**: All documentation files use snake_case (underscores between words) and have a `.md` extension
2. **Directory Names**: Directory names are lowercase with no separators
3. **Document Headers**: Each document should have a header section with type, last updated date, and related documents
4. **Cross-References**: Documents should include cross-references to related documents

For more details, see:
- [Naming Conventions](./naming_conventions.md)
- [Documentation Types and Organization](./documentation_types.md)
- [Cross-Referencing Guide](./cross_referencing_guide.md)

## Maintaining Documentation

When adding or updating documentation:

1. Follow the naming conventions and documentation standards
2. Update the documentation index when adding new documents
3. Add cross-references to related documents
4. Verify cross-references using the verification script:
   ```
   python scripts/check_links.py
   ```
5. Update the "Last Updated" date in the document header

## Project Status

For the current project status, see the [Project Status Dashboard](./project/project_status_dashboard.md).

## Documentation Issues or Questions

If you have questions or issues with the documentation, please reach out to the documentation owner (@username).
