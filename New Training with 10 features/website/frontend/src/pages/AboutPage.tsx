import { Code2, Database, FlaskConical, Users } from "lucide-react";

type AboutPageProps = {
  modelCount: number;
  datasetCount: number;
};

const technologies = ["Python", "Librosa", "Scikit-learn", "XGBoost", "Qiskit", "FastAPI", "React", "Vite", "SHAP"];

export function AboutPage({ modelCount, datasetCount }: AboutPageProps) {
  return (
    <div className="page-stack about-page">
      <header className="page-heading">
        <div>
          <h1>About The Project</h1>
          <p>
            This top-10 feature MVP supports Parkinson's speech research by combining audio feature extraction,
            reduced-feature prediction, model comparison, and visual explainability.
          </p>
        </div>
      </header>

      <section className="about-grid">
        <article className="panel about-panel">
          <FlaskConical size={24} />
          <h2>Project Description</h2>
          <p>
            The application extracts the selected 10 speech features from audio, evaluates them with classical and
            hybrid quantum models, then presents prediction results with grouped explanations.
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
          <ul className="team-list">
            <li>Karthikeyan Bharadwaj S</li>
            <li>Naga Lahari T</li>
            <li>Sai Kushal V</li>
          </ul>
        </article>
      </section>
    </div>
  );
}
