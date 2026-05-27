Contributing Guidelines

Thank you for considering contributing to this project. These guidelines will help keep contributions consistent and enable fast reviews.

How to contribute

- Fork the repository and create a feature branch: `git checkout -b feature/your-feature`.
- Make small, focused commits with clear messages.
- Run tests and linters locally before opening a PR.
- Open a Pull Request describing the problem, your approach, and any trade-offs.

Coding standards

- Follow existing code style. Use `black` and `isort` for formatting:

```bash
black .
isort .
```

- Write clear, minimal functions and add unit tests for new behavior.

Tests & CI

- Add tests to `test/` and run them with `pytest -q`.
- Keep test fixtures deterministic and fast where possible.

Reviews

- Assign reviewers and respond to feedback promptly.
- Ensure CI passes before merging.

Issue tracking

- Create issues for bugs or feature requests and reference them in PRs.

License

- Your contributions will be made under the repository's license (add a LICENSE file if none exists).

Code of Conduct

- Be respectful and collaborative in discussions. Open an issue if you encounter conflicts.
