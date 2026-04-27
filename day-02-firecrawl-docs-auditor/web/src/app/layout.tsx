import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Firecrawl Docs Auditor",
  description: "Local Day 2 scaffold for an agent-native docs audit.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
