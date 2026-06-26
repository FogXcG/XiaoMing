---
name: skill-installer
description: Install skills from GitHub URLs into this workspace. Use when the user asks to install a skill, provides a skill URL, or wants to add a remote skill.
---

# Skill Installer

Use this skill when the user wants to install an Agent Skill from a GitHub directory URL or repo/path.

Supported input:

- GitHub tree URLs such as `https://github.com/<owner>/<repo>/tree/<ref>/<path-to-skill>`
- GitHub repo plus one or more paths, such as repo `<owner>/<repo>` and path `skills/<skill-name>`
- Optional destination directory. Default is this workspace's `.agents/skills`.

Decision workflow:

- If the user provided a GitHub tree URL, call `skill` with `action="install"` and `url`.
- If the user provided repo plus path, call `skill` with `action="install"`, `repo`, and `paths`. `paths` must be an array, for example `["skills/using-superpowers"]`; never pass a comma-separated string.
- If the user provided only a skill name, do not guess the source. Use `skill(action="load", name="find-skills")` to discover candidates, or call `talk` to ask the coordinator for the GitHub URL or repo/path.
- Only use OpenAI skills as a source when the user explicitly asks for OpenAI curated, experimental, or system skills.

Interactive chat workflow:

1. Explain that remote skill installation changes the local workspace.
2. Call `schedule_background_task` with the installation request so a background worker performs the installation.
3. Tell the user they can continue chatting while the worker installs the skill.

Worker or direct command workflow:

1. Explain that remote skill installation changes the local workspace.
2. Call the `skill` tool with `action="install"` and either `url`, or `repo` plus `paths`. Use `ref` or `dest` only when needed.
3. If installation succeeds, tell the user the skill is available locally. To use it now, load it with `skill(action="load", name=...)`.
4. If the result is `already_installed`, do not reinstall or overwrite it. Load the skill when you need to follow its instructions.

Do not run `npx skills add`, especially with `-g`. The native installer writes the skill to the intended destination and refreshes the current skill library.
Do not manually install a skill with `curl`, `wget`, `git clone`, `mkdir`, `cp`, `write_file`, or other file tools. If `skill(action="install")` fails after a valid source/path, report the failure to the coordinator instead of recreating the installer by hand.
Do not repeatedly `web_search` generic repositories when the skill source is unknown. Use `find-skills` or ask the coordinator for the source.
Do not run scripts from installed skills during installation. The installer only downloads files into the destination skill directory.
