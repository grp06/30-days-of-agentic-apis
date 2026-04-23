type DiffViewerProps = {
  diffText: string | null;
};

export function DiffViewer({ diffText }: DiffViewerProps) {
  if (!diffText) {
    return <p className="muted">Select a diff card to inspect the patch.</p>;
  }
  return (
    <pre className="diff-viewer">
      <code>{diffText}</code>
    </pre>
  );
}
