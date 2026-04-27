import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Providers from "@/components/Providers";

export const metadata: Metadata = {
  title: "SDP Voice Lab V2",
  description:
    "A polished Parkinson's speech-feature research dashboard with configurable model labels, strict external CSV testing, and explainable predictions.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <Providers>
          <Navbar />
          <main className="relative pb-16">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
