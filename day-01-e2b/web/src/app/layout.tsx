import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Agent Black Box",
  description: "Replay and fork AI coding runs recorded in E2B sandboxes.",
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
