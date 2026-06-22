---
name: find-skills
description: Helps users discover and install agent skills when they ask questions like "how do I do X", "find a skill for X", "is there a skill that can...", or express interest in extending capabilities. This skill should be used when the user is looking for functionality that might exist as an installable skill.
---

# Find Skills

This skill helps you discover and load skills from the open agent skills ecosystem.

## When to Use This Skill

Use this skill when the user:

- Asks "how do I do X" where X might be a common task with an existing skill
- Says "find a skill for X" or "is there a skill for X"
- Asks "can you do X" where X is a specialized capability
- Expresses interest in extending agent capabilities
- Wants to search for tools, templates, or workflows
- Mentions they wish they had help with a specific domain (design, testing, deployment, etc.)

## Skill Ecosystem

Skills are modular packages that extend agent capabilities with specialized knowledge, workflows, and tools. Skills are published on GitHub with a `SKILL.md` file.

**Browse skills at:** https://skills.sh/

## Workflow

### Step 1: Understand What They Need

Identify:
1. The domain (e.g., React, testing, design, deployment)
2. The specific task (e.g., writing tests, creating animations, reviewing PRs)
3. Whether this is a common enough task that a skill likely exists

### Step 2: Search for Skills

Use `web_search` to find skills. Example queries:

- `skills.sh react best practices`
- `github.com SKILL.md react component patterns`
- For popular sources: `vercel-labs/agent-skills skills`
- Check https://skills.sh/ leaderboard for top skills

If the local `skills` CLI is available, `npx skills find <query>` can be used as a discovery aid only. Do not install with the CLI; use the native skill tools in the following steps.

### Step 3: Verify Quality

Before recommending a skill:

1. **Install count** — Prefer skills with 1K+ installs. Be cautious with anything under 100.
2. **Source reputation** — Official sources (`vercel-labs`, `anthropics`, `microsoft`) are more trustworthy than unknown authors.
3. **GitHub stars** — A skill from a repo with <100 stars should be treated with skepticism.

### Step 4: Check Local Cache First

Before downloading, check if the skill is already installed locally:

- Use `list_files` to check `.agents/skills/{name}/SKILL.md`
- If cached → use `load_skill` with the skill name to load it into the session
- If not → use `fetch_skill` with the GitHub tree URL to download and load it

### Step 5: Present to User

When you find relevant skills, present them with:

1. The skill name and what it does
2. The install count and source
3. The GitHub URL needed for fetch_skill

Example:

```text
I found a skill that might help! The "react-best-practices" skill provides
React and Next.js performance optimization guidelines from Vercel Engineering.
(185K installs)

It's ready to fetch. Shall I load it?
```

### Step 6: Load the Skill

If the user approves or the match is clear:

- If not cached → `fetch_skill(url)` downloads, installs, and loads the skill into this session
- If already cached in `.agents/skills/` → `load_skill(name)` loads it directly

For skills the user wants to keep permanently, you can also suggest `install_skill`.

## Common Skill Categories

| Category        | Example Queries                          |
| --------------- | ---------------------------------------- |
| Web Development | react, nextjs, typescript, css, tailwind |
| Testing         | testing, jest, playwright, e2e           |
| DevOps          | deploy, docker, kubernetes, ci-cd        |
| Documentation   | docs, readme, changelog, api-docs        |
| Code Quality    | review, lint, refactor, best-practices   |
| Design          | ui, ux, design-system, accessibility     |
| Productivity    | workflow, automation, git                |

## Tips for Effective Searches

1. **Use specific keywords**: "react testing" is better than just "testing"
2. **Try alternative terms**: If "deploy" doesn't work, try "deployment" or "ci-cd"
3. **Check popular sources**: Many skills come from `vercel-labs/agent-skills` or `ComposioHQ/awesome-claude-skills`

## When No Skills Are Found

If no relevant skills exist:

1. Acknowledge that no existing skill was found
2. Offer to help with the task directly using your general capabilities
3. Suggest the user could create their own skill
