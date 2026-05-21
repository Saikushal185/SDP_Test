import { Code2, Database, FlaskConical, Users } from "lucide-react";

type AboutPageProps = {
  modelCount: number;
  datasetCount: number;
};

const technologies = ["Python", "Scikit-learn", "XGBoost", "Qiskit", "FastAPI", "React", "Vite", "SHAP"];

export function AboutPage({ modelCount, datasetCount }: AboutPageProps) {
  return (
    <div className="page-stack about-page">
      <header className="page-heading">
        <div>
          <h1>About The Project</h1>
          <p>
            This MVP supports early Parkinson's research by combining speech-feature prediction, model comparison,
            and visual explainability in one healthcare-focused interface.
          </p>
        </div>
      </header>

      <section className="about-grid">
        <article className="panel about-panel">
          <FlaskConical size={24} />
          <h2>Project Description</h2>
          <p>
            The application evaluates speech-derived features with classical and hybrid quantum models, then presents
            prediction results with grouped SHAP explanations for clearer interpretation.
          </p>
        </article>

        <article className="panel about-panel">
          <Database size={24} />
          <h2>Research Assets</h2>
          <p>
            The current workspace includes {datasetCount} dataset group{datasetCount === 1 ? "" : "s"} and{" "}
            {modelCount} saved model artifact{modelCount === 1 ? "" : "s"} for local analysis.
          </p>
        </article>

        <article className="panel about-panel tech-panel">
          <Code2 size={24} />
          <h2>Technologies Used</h2>
          <div className="tech-list">
            {technologies.map((technology) => (
              <span key={technology}>{technology}</span>
            ))}
          </div>
        </article>

        <article className="panel about-panel">
          <Users size={24} />
          <h2>Team Details</h2>
          <p>
            Add team member names, roles, advisor details, and institution information here before final submission.
          </p>
        </article>
      </section>
    </div>
  );
}
