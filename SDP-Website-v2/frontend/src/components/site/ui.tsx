import Link from "next/link";
import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from "react";

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}

export function SiteContainer({
  children,
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cx(
        "mx-auto w-full max-w-7xl px-4 sm:px-6 lg:px-8",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function SectionHeading({
  eyebrow,
  title,
  description,
  className,
  align = "left",
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  className?: string;
  align?: "left" | "center";
}) {
  return (
    <div className={cx(className, align === "center" && "text-center")}>
      {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
      <h2 className="section-title">{title}</h2>
      {description ? <p className="section-description">{description}</p> : null}
    </div>
  );
}

export function PageIntro({
  eyebrow,
  title,
  description,
  actions,
  meta,
}: {
  eyebrow?: string;
  title: string;
  description: string;
  actions?: ReactNode;
  meta?: Array<{ label: string; value: string }>;
}) {
  return (
    <section className="pt-10 sm:pt-14">
      <SiteContainer>
        <div className="grid gap-8 border-b border-[var(--border-subtle)] pb-8 lg:grid-cols-[minmax(0,1fr)_340px] lg:items-end">
          <div className="max-w-3xl">
            {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
            <h1 className="page-title">{title}</h1>
            <p className="page-subtitle">{description}</p>
            {actions ? <div className="mt-6 flex flex-wrap gap-3">{actions}</div> : null}
          </div>

          {meta?.length ? (
            <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
              {meta.map((item) => (
                <div key={item.label} className="site-panel">
                  <p className="text-[0.72rem] font-semibold uppercase tracking-[0.22em] text-[var(--text-muted)]">
                    {item.label}
                  </p>
                  <p className="mt-2 text-xl font-semibold text-[var(--text-strong)]">
                    {item.value}
                  </p>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      </SiteContainer>
    </section>
  );
}

export function Panel({
  children,
  className,
  tone = "default",
}: {
  children: ReactNode;
  className?: string;
  tone?: "default" | "muted" | "strong";
}) {
  return (
    <section
      className={cx(
        tone === "default" && "site-panel",
        tone === "muted" && "site-panel-muted",
        tone === "strong" && "site-panel-strong",
        className
      )}
    >
      {children}
    </section>
  );
}

export function StatusPill({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: "neutral" | "accent" | "positive" | "caution" | "critical";
  className?: string;
}) {
  return (
    <span
      className={cx(
        "status-pill",
        tone === "accent" && "status-pill-accent",
        tone === "positive" && "status-pill-positive",
        tone === "caution" && "status-pill-caution",
        tone === "critical" && "status-pill-critical",
        className
      )}
    >
      {children}
    </span>
  );
}

export function MetricCard({
  label,
  value,
  description,
  accent = "default",
}: {
  label: string;
  value: string;
  description: string;
  accent?: "default" | "accent";
}) {
  return (
    <div className={accent === "accent" ? "site-panel-strong" : "site-panel"}>
      <p className="text-[0.76rem] font-semibold uppercase tracking-[0.2em] text-[var(--text-muted)]">
        {label}
      </p>
      <p className="metric-value mt-3">{value}</p>
      <p className="mt-3 text-sm leading-6 text-[var(--text-muted)]">
        {description}
      </p>
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon: ReactNode;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <Panel className="flex min-h-[320px] flex-col items-center justify-center text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-full border border-[var(--border-strong)] bg-[var(--surface-alt)] text-2xl text-[var(--accent-strong)]">
        {icon}
      </div>
      <h2 className="mt-6 text-2xl font-semibold text-[var(--text-strong)]">
        {title}
      </h2>
      <p className="mt-3 max-w-md text-sm leading-7 text-[var(--text-muted)]">
        {description}
      </p>
      {action ? <div className="mt-6">{action}</div> : null}
    </Panel>
  );
}

export function buttonClasses(variant: "primary" | "secondary" | "ghost" = "primary") {
  return cx(
    "inline-flex items-center justify-center rounded-full px-5 py-3 text-sm font-semibold transition duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-transparent disabled:cursor-not-allowed disabled:opacity-60",
    variant === "primary" && "button-primary",
    variant === "secondary" && "button-secondary",
    variant === "ghost" && "button-ghost"
  );
}

export function LinkButton({
  href,
  children,
  className,
  variant = "primary",
}: {
  href: string;
  children: ReactNode;
  className?: string;
  variant?: "primary" | "secondary" | "ghost";
}) {
  return (
    <Link href={href} className={cx(buttonClasses(variant), className)}>
      {children}
    </Link>
  );
}

export function ActionButton({
  children,
  className,
  variant = "primary",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
}) {
  return (
    <button className={cx(buttonClasses(variant), className)} {...props}>
      {children}
    </button>
  );
}

export { cx };
