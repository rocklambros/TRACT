# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| main    | Yes                |
| < main  | No                 |

Only the latest code on the `main` branch receives security updates.

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

To report a vulnerability, email **rock@rockcyber.ai** with:

1. A description of the vulnerability
2. Steps to reproduce (or a proof-of-concept)
3. The impact you've assessed
4. Any suggested fix (optional)

You will receive an acknowledgment within 48 hours. We aim to provide a substantive response (fix, mitigation, or explanation) within 7 days.

## Scope

The following are in scope:

- Code in `tract/`, `scripts/`, and `parsers/`
- Dependencies listed in `requirements.txt`
- GitHub Actions workflows in `.github/workflows/`
- Data processing pipelines that handle external input (OpenCRE API, framework source files)

The following are out of scope:

- The CRE data itself (report to [opencre.org](https://opencre.org))
- Framework source documents (report to the respective framework maintainers)
- Vulnerabilities in third-party dependencies (report upstream; we will update pinned versions)

## Disclosure Policy

We follow coordinated disclosure. We will:

1. Confirm the issue and determine its impact
2. Develop and test a fix
3. Release the fix and credit the reporter (unless anonymity is requested)
4. Publish a brief advisory if the issue affected released artifacts

## Security Controls

This project enforces several security practices:

- All dependencies are pinned to exact versions in `requirements.txt`
- Dependabot monitors for vulnerable dependencies weekly
- CodeQL scans run on every push to `main` and on PRs
- Secret scanning and push protection are enabled on the repository
- No `eval()`, `exec()`, `subprocess(shell=True)`, or `pickle` on untrusted data
- All external text is sanitized (null bytes, HTML, zero-width characters) before storage
- API credentials are never hardcoded — sourced from environment variables or `pass` manager
