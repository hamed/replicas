# Releasing

Releases are built on GitHub and uploaded with PyPI Trusted Publishing. No
long-lived PyPI token belongs in the repository.

## One-time setup

1. Create a pending publisher at <https://pypi.org/manage/account/publishing/>
   with these values:

   - PyPI project: `replicas`
   - GitHub owner: `hamed`
   - repository: `replicas`
   - workflow: `release.yml`
   - environment: `pypi`

2. Create a GitHub environment named `pypi` and require manual approval for
   deployments to it. The environment name must match the pending publisher.

The pending publisher creates the PyPI project on the first successful upload.

## Publishing a release

1. Update `project.version` in `pyproject.toml` and move the release notes from
   `Unreleased` to a dated section in `CHANGELOG.md`.
2. Merge the release PR and wait for the test workflow on `main` to pass.
3. Publish a GitHub Release targeting `main`. Its tag must be exactly
   `v<project.version>`, for example `v0.1.0`.
4. Approve the `pypi` deployment. The release workflow verifies the tag,
   builds and checks both distributions, then publishes them to PyPI.
5. Verify the release from a clean environment with
   `pip install 'replicas[pandas]'`.

PyPI files are immutable. Never reuse a version after any file for it has been
accepted; fix the problem and publish a new patch release instead.
