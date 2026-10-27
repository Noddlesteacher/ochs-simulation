# Author checklist for a GitHub release

Current preparation: 26 October 2026. The complete reproduction has passed,
and its evidence is included in `validation/`. This checklist covers the
remaining publication process; the repository has not yet been published.

## Before publication

- Read README.md and REVIEWER_GUIDE.md as a new reader would.
- Confirm that START_HERE.py is the only default command advertised.
- Retain the completed full from-seed run and its interruption/recovery evidence.
- Retain FULL_REPRODUCTION_VERIFIED.json and the raw, summary, table and figure
  comparison records; do not manufacture that success marker manually.
- Check the exact manuscript version: four active figures and seven tables.
- Decide a software licence with the authors. Public visibility alone does not
  grant a licence to reuse the code.
- Review author names, citation instructions and any journal-specific anonymity
  requirements before making the repository public.
- Ensure no original raw experimental data are required as hidden inputs.
- Do not upload virtual environments, caches, local raw-run outputs or old ZIPs.
- Tag the reviewed commit, record its SHA, and freeze the submitted version.

## What a repository visitor should be able to answer

1. What experiment is reproduced and how large is it?
2. Which Python and dependency versions do I need?
3. Which single command starts the complete reproduction?
4. How long and how much disk/memory should I allow?
5. Where do generated data, predictions, tables and figures go?
6. Which function corresponds to each part of the method?
7. Which checks have actually run, and what remains unverified?

There is no need for a reviewer to reverse-engineer the historical research
directory. The public pipeline and the preserved exact numerical backend have
separate explanations.
