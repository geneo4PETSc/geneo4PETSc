If you wish to contribute, please follow these guidelines:

## Code style / code rules

- Use Camel code style. If you speak regex, this means basically: `[a-z]*([A-Z][a-z]*)*`. Otherwise, this means: `useThisStyle` instead of `use_this_style`.
- Space: do not add extra spaces.
- Indentation: do not insert tabulations (replace tabulations with 2 spaces).
- Try, as much as possible, to integrate your contribution in a consistent manner into the existing code base (style, use same variable names when possible, use const when possible, ...).
- When committing, try to tag your commits when relevant ([BUG FIX], [CLEAN], ...): this is helpful for maintainers.

## Steps for creating good issues

- Reduce the problem as much as possible (reduce input data).
- Reduce the code (triggering an issue) as much as possible.
- Reduce the steps to reproduce the issue as much as possible.

## Steps for creating good pull requests

- Create a dedicated branch for each pull request (do not pull request on master).
- Once the pull request is validated on its dedicated branch, it will be rebased on master.
