import type { ReactNode } from "react";

type MarkdownTextProps = {
  content: string;
};

type Block =
  | { type: "heading"; level: number; text: string }
  | { type: "paragraph"; text: string }
  | { type: "ordered-list"; items: string[] }
  | { type: "unordered-list"; items: string[] }
  | { type: "code"; code: string };

export function MarkdownText({ content }: MarkdownTextProps) {
  const normalized = content.replace(/\r\n/g, "\n").trim();

  if (!normalized) {
    return null;
  }

  const blocks = parseBlocks(normalized);

  return (
    <div className="markdown-text">
      {blocks.map((block, index) => renderBlock(block, index))}
    </div>
  );
}

function parseBlocks(content: string): Block[] {
  const lines = content.split("\n");
  const blocks: Block[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index].trimEnd();
    const trimmed = line.trim();

    if (!trimmed) {
      index += 1;
      continue;
    }

    if (trimmed.startsWith("```")) {
      const codeLines: string[] = [];
      index += 1;

      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }

      if (index < lines.length) {
        index += 1;
      }

      blocks.push({ type: "code", code: codeLines.join("\n") });
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      blocks.push({
        type: "heading",
        level: headingMatch[1].length,
        text: headingMatch[2].trim(),
      });
      index += 1;
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      const items: string[] = [];
      while (index < lines.length) {
        const candidate = lines[index].trim();
        const match = candidate.match(/^\d+\.\s+(.*)$/);
        if (!match) {
          break;
        }
        items.push(match[1].trim());
        index += 1;
      }
      blocks.push({ type: "ordered-list", items });
      continue;
    }

    if (/^[-*+]\s+/.test(trimmed)) {
      const items: string[] = [];
      while (index < lines.length) {
        const candidate = lines[index].trim();
        const match = candidate.match(/^[-*+]\s+(.*)$/);
        if (!match) {
          break;
        }
        items.push(match[1].trim());
        index += 1;
      }
      blocks.push({ type: "unordered-list", items });
      continue;
    }

    const paragraphLines: string[] = [];
    while (index < lines.length) {
      const candidate = lines[index];
      const candidateTrimmed = candidate.trim();
      if (!candidateTrimmed) {
        index += 1;
        break;
      }
      if (
        candidateTrimmed.startsWith("```") ||
        /^(#{1,6})\s+/.test(candidateTrimmed) ||
        /^\d+\.\s+/.test(candidateTrimmed) ||
        /^[-*+]\s+/.test(candidateTrimmed)
      ) {
        break;
      }

      paragraphLines.push(candidateTrimmed);
      index += 1;
    }

    blocks.push({
      type: "paragraph",
      text: paragraphLines.join(" "),
    });
  }

  return blocks;
}

function renderBlock(block: Block, index: number) {
  if (block.type === "heading") {
    const headingContent = renderInline(block.text);
    if (block.level === 1) {
      return <h1 key={index}>{headingContent}</h1>;
    }
    if (block.level === 2) {
      return <h2 key={index}>{headingContent}</h2>;
    }
    if (block.level === 3) {
      return <h3 key={index}>{headingContent}</h3>;
    }
    if (block.level === 4) {
      return <h4 key={index}>{headingContent}</h4>;
    }
    if (block.level === 5) {
      return <h5 key={index}>{headingContent}</h5>;
    }
    return <h6 key={index}>{headingContent}</h6>;
  }

  if (block.type === "paragraph") {
    return <p key={index}>{renderInline(block.text)}</p>;
  }

  if (block.type === "ordered-list") {
    return (
      <ol key={index}>
        {block.items.map((item, itemIndex) => (
          <li key={itemIndex}>{renderInline(item)}</li>
        ))}
      </ol>
    );
  }

  if (block.type === "unordered-list") {
    return (
      <ul key={index}>
        {block.items.map((item, itemIndex) => (
          <li key={itemIndex}>{renderInline(item)}</li>
        ))}
      </ul>
    );
  }

  return (
    <pre key={index}>
      <code>{block.code}</code>
    </pre>
  );
}

function renderInline(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  let lastIndex = 0;
  let key = 0;

  for (const match of text.matchAll(pattern)) {
    const matchedText = match[0];
    const start = match.index ?? 0;

    if (start > lastIndex) {
      nodes.push(text.slice(lastIndex, start));
    }

    if (matchedText.startsWith("**") && matchedText.endsWith("**")) {
      nodes.push(
        <strong key={key}>{matchedText.slice(2, -2)}</strong>,
      );
      key += 1;
      lastIndex = start + matchedText.length;
      continue;
    }

    if (matchedText.startsWith("`") && matchedText.endsWith("`")) {
      nodes.push(
        <code key={key}>{matchedText.slice(1, -1)}</code>,
      );
      key += 1;
      lastIndex = start + matchedText.length;
      continue;
    }

    const linkMatch = matchedText.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (linkMatch) {
      nodes.push(
        <a
          key={key}
          href={linkMatch[2]}
          target="_blank"
          rel="noreferrer"
        >
          {linkMatch[1]}
        </a>,
      );
      key += 1;
      lastIndex = start + matchedText.length;
      continue;
    }
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }

  return nodes;
}
