"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { FaBars, FaDatabase, FaTimes } from "react-icons/fa";
import { cx, SiteContainer, StatusPill } from "@/components/site/ui";

const navItems = [
  { name: "Overview", path: "/" },
  { name: "Analysis", path: "/upload" },
  { name: "Dataset Test", path: "/dataset-test" },
  { name: "Prediction", path: "/prediction" },
  { name: "Explainability", path: "/explainability" },
  { name: "Performance", path: "/performance" },
];

export default function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  const isActive = (path: string) =>
    path === "/" ? pathname === "/" : pathname === path || pathname.startsWith(`${path}/`);

  return (
    <nav className="sticky top-0 z-50 border-b border-[rgba(10,35,40,0.1)] bg-[rgba(243,238,226,0.82)] backdrop-blur-2xl">
      <SiteContainer>
        <div className="flex min-h-[72px] items-center justify-between gap-3">
          <Link href="/" className="group flex min-w-0 items-center gap-3 pr-2">
            <span className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-[var(--accent-strong)] text-[#fff8ea] shadow-[0_12px_30px_rgba(15,55,61,0.18)]">
              <FaDatabase />
            </span>
            <span>
              <span className="block font-display text-xl leading-none text-[var(--text-strong)]">
                SDP Voice Lab
              </span>
              <span className="mt-1 hidden text-xs font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)] sm:block">
                one-dataset model registry
              </span>
            </span>
          </Link>

          <div className="hidden items-center gap-1 xl:flex">
            {navItems.map((item) => (
              <Link
                key={item.path}
                href={item.path}
                className={cx(
                  "rounded-full px-4 py-2 text-sm font-semibold transition",
                  isActive(item.path)
                    ? "bg-[var(--accent-strong)] text-[#fff8ea] shadow-[0_10px_24px_rgba(15,55,61,0.16)]"
                    : "text-[var(--text-muted)] hover:bg-white/70 hover:text-[var(--text-strong)]"
                )}
              >
                {item.name}
              </Link>
            ))}
          </div>

          <div className="hidden lg:block xl:ml-2">
            <StatusPill tone="positive">Strict CSV evaluation</StatusPill>
          </div>

          <button
            className="rounded-full border border-[var(--border-subtle)] bg-white/70 p-2.5 text-[var(--text-strong)] xl:hidden"
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label="Toggle navigation"
          >
            {mobileOpen ? <FaTimes size={22} /> : <FaBars size={22} />}
          </button>
        </div>
      </SiteContainer>

      {mobileOpen ? (
        <div className="border-t border-[var(--border-subtle)] bg-[rgba(248,245,239,0.96)] xl:hidden">
          <SiteContainer className="space-y-2 py-4">
            {navItems.map((item) => (
              <Link
                key={item.path}
                href={item.path}
                onClick={() => setMobileOpen(false)}
                className={cx(
                  "block rounded-2xl px-4 py-3 text-sm font-semibold transition",
                  isActive(item.path)
                    ? "bg-[var(--accent-strong)] text-[#fff8ea]"
                    : "bg-white/62 text-[var(--text-muted)]"
                )}
              >
                {item.name}
              </Link>
            ))}
            <div className="pt-2">
              <StatusPill tone="positive">Strict CSV evaluation</StatusPill>
            </div>
          </SiteContainer>
        </div>
      ) : null}
    </nav>
  );
}
