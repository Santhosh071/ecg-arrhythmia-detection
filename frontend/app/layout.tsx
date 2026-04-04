import type { Metadata } from "next";
import "@/app/globals.css";

export const metadata: Metadata = {
  title: "ECG Arrhythmia Monitor",
  description: "Production frontend for ECG arrhythmia monitoring and clinical review."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
