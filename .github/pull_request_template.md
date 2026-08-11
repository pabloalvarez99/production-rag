## Multi-agent checklist

- [ ] Authored by **A1 core / A2 docs / A3 glue** (select one), and all changes stay within that seat's owned paths.
- [ ] Merge order is respected: no glue change references a symbol that is not yet tracked in Git.
- [ ] The offline suite is green and the credential-free path remains unchanged.
- [ ] The diff contains no credential value.
- [ ] Any config key whose status changed is commented `LIVE` or `DECLARED`, with the reason.
- [ ] If a README claim changed, `tests/test_readme_claims.py` passes.
