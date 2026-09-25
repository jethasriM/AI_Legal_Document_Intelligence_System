import { useState } from "react";
import "./index.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [activePage, setActivePage] = useState("Analyze");

  const analyzeDocument = async () => {
    if (!file) {
      setError("Please select a document first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Analysis failed.");
      }

      console.log("LexTrace API response:", data);

      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  /*
   * IMPORTANT:
   * FastAPI returns:
   *
   * {
   *   document: {...},
   *   draft: {...},
   *   sections: [...],
   *   grounding: ...
   * }
   *
   * So we use document and draft, NOT documents[0] / drafts[0].
   */
  const documentData = result?.document;
  const draftData = result?.draft;

  return (
    <div className="app-shell">

      {/* =========================================================
          SIDEBAR
      ========================================================= */}

      <aside className="sidebar">

        <div className="brand">
          <div className="brand-title">⚖ LexTrace AI</div>

          <div className="brand-subtitle">
            Evidence-grounded legal document intelligence
          </div>
        </div>

        <div className="sidebar-label">
          Workflow
        </div>

        <nav>
          {["Analyze", "Documents", "Review", "Evaluation"].map((page) => (
            <button
              key={page}
              type="button"
              className={`nav-item ${
                activePage === page ? "active" : ""
              }`}
              onClick={() => setActivePage(page)}
            >
              {page}
            </button>
          ))}
        </nav>

        <div className="sidebar-label">
          System
        </div>

        <div className="capability">
          ✓ OCR-aware ingestion
        </div>

        <div className="capability">
          ✓ Structured extraction
        </div>

        <div className="capability">
          ✓ Hybrid retrieval
        </div>

        <div className="capability">
          ✓ Grounded generation
        </div>

        <div className="capability">
          ✓ Page-level evidence
        </div>

        <div className="capability">
          ✓ Human review
        </div>

        <div className="system-card">
          <div className="system-status">
            <span className="status-dot"></span>
            System ready
          </div>

          <div className="system-detail">
            Evidence tracing enabled
          </div>
        </div>

      </aside>


      {/* =========================================================
          MAIN CONTENT
      ========================================================= */}

      <main className="main-content">

        {/* =======================================================
            ANALYZE PAGE
        ======================================================= */}

        {activePage === "Analyze" && (
          <>
            <section className="hero">

              <div className="eyebrow">
                Evidence-grounded document intelligence
              </div>

              <h1>
                LexTrace <span>AI</span>
              </h1>

              <p>
                Understand messy legal documents, extract structured facts,
                retrieve supporting evidence, and generate reviewable drafts
                without losing the connection to the source.
              </p>

              <div className="hero-note">
                ● Source evidence remains visible throughout the workflow
              </div>

            </section>


            {/* ---------------------------------------------------
                INTAKE
            --------------------------------------------------- */}

            <div className="section-kicker">
              01 · Intake
            </div>

            <h2>
              Start an analysis
            </h2>

            <p className="section-copy">
              Upload a legal-style document and let the LexTrace pipeline
              process it.
            </p>

            <section className="upload-card">

              <label className="upload-zone">

                <input
                  type="file"
                  accept=".txt,.pdf,.png,.jpg,.jpeg,.tiff"
                  onChange={(e) => {
                    setFile(e.target.files[0]);
                    setError("");
                    setResult(null);
                  }}
                />

                <div className="upload-icon">
                  ↑
                </div>

                <div className="upload-title">
                  {file
                    ? file.name
                    : "Drop a document here"}
                </div>

                <div className="upload-description">
                  TXT, PDF, PNG, JPG or TIFF
                </div>

              </label>


              {file && (
                <div className="selected-file">

                  <div>
                    <strong>
                      {file.name}
                    </strong>

                    <span>
                      {(file.size / 1024).toFixed(1)} KB
                    </span>
                  </div>

                  <button
                    type="button"
                    className="analyze-button"
                    onClick={analyzeDocument}
                    disabled={loading}
                  >
                    {loading
                      ? "Analyzing..."
                      : "Analyze document"}
                  </button>

                </div>
              )}


              {error && (
                <div className="error-message">
                  {error}
                </div>
              )}

            </section>


            {/* ---------------------------------------------------
                RESULTS
            --------------------------------------------------- */}

            {documentData && (
              <>
                <div className="section-kicker">
                  02 · Intelligence
                </div>

                <h2>
                  Document intelligence
                </h2>

                <p className="section-copy">
                  A structured view of what was extracted, what remains
                  uncertain, and what the draft used as evidence.
                </p>


                {/* METRICS */}

                <div className="metrics">

                  <Metric
                    label="Document"
                    value={formatDocType(
                      documentData.doc_type
                    )}
                    detail="Detected document type"
                  />

                  <Metric
                    label="Source quality"
                    value={`${documentData.confidence_flags?.quality_score ?? 0}/100`}
                    detail="Extraction quality"
                  />

                  <Metric
                    label="Pages"
                    value={
                      documentData.page_count ?? 1
                    }
                    detail="Detected pages"
                  />

                  <Metric
                    label="Fields"
                    value={
                      Object.keys(
                        documentData.structured_fields || {}
                      ).length
                    }
                    detail="Structured fields"
                  />

                </div>


                {/* ------------------------------------------------
                    SOURCE DOCUMENT
                ------------------------------------------------ */}

                <section className="content-section">

                  <div className="section-header">

                    <h3>
                      Source document
                    </h3>

                    <span>
                      {documentData.source_path}
                    </span>

                  </div>

                  <div className="source-viewer">
                    {documentData.cleaned_text}
                  </div>

                </section>


                {/* ------------------------------------------------
                    EXTRACTED INTELLIGENCE
                ------------------------------------------------ */}

                <section className="content-section">

                  <div className="section-header">

                    <h3>
                      Extracted intelligence
                    </h3>

                  </div>

                  <div className="field-grid">

                    {Object.entries(
                      documentData.structured_fields || {}
                    ).map(([key, value]) => {

                      const evidence =
                        documentData.field_evidence?.[key];

                      const status =
                        evidence?.status || "verified";

                      return (
                        <div
                          className="field-card"
                          key={key}
                        >

                          <div className="field-name">

                            {formatFieldName(key)}

                            <span
                              className={`badge ${status}`}
                            >
                              {status}
                            </span>

                          </div>

                          <div className="field-value">
                            {formatValue(value)}
                          </div>

                          {evidence?.page && (
                            <div className="field-evidence">
                              Page {evidence.page}
                            </div>
                          )}

                          {evidence?.excerpt && (
                            <div className="evidence-excerpt">
                              "{evidence.excerpt}"
                            </div>
                          )}

                        </div>
                      );
                    })}

                  </div>

                </section>


                {/* ------------------------------------------------
                    GROUNDED DRAFT
                ------------------------------------------------ */}

                {draftData && (
                  <section className="content-section">

                    <div className="section-header">

                      <h3>
                        Grounded draft
                      </h3>

                      {draftData.overall_grounding_score !==
                        undefined && (
                        <span className="grounding">
                          Model-reported grounding:{" "}
                          {Math.round(
                            draftData.overall_grounding_score *
                              100
                          )}
                          %
                        </span>
                      )}

                    </div>


                    <div className="draft">

                      {draftData.sections?.map(
                        (section, index) => (

                          <div
                            className="draft-section"
                            key={index}
                          >

                            <div className="draft-section-header">

                              <h4>
                                {section.section_name}
                              </h4>

                              {section.grounding_score !==
                                undefined && (
                                <span className="section-grounding">
                                  {Math.round(
                                    section.grounding_score *
                                      100
                                  )}
                                  % supported
                                </span>
                              )}

                            </div>


                            <p>
                              {section.content ||
                                "[Section not generated]"}
                            </p>


                            {section.supporting_passages
                              ?.length > 0 && (

                              <div className="draft-evidence">

                                <div className="draft-evidence-label">
                                  Source evidence
                                </div>

                                {section.supporting_passages.map(
                                  (
                                    passage,
                                    passageIndex
                                  ) => (

                                    <div
                                      className="draft-evidence-item"
                                      key={passageIndex}
                                    >

                                      <span className="draft-evidence-id">
                                        {passage.passage_id}
                                      </span>

                                      <span>
                                        {passage.excerpt}
                                      </span>

                                      <span className="draft-evidence-score">
                                        {passage.score !==
                                        undefined
                                          ? passage.score.toFixed(
                                              2
                                            )
                                          : "—"}
                                      </span>

                                    </div>

                                  )
                                )}

                              </div>
                            )}

                          </div>

                        )
                      )}

                    </div>

                  </section>
                )}


                {/* ------------------------------------------------
                    WARNINGS
                ------------------------------------------------ */}

                {documentData.warnings?.length > 0 && (
                  <section className="content-section">

                    <div className="warning-box">

                      <strong>
                        Extraction warnings
                      </strong>

                      {documentData.warnings.map(
                        (warning, index) => (
                          <div key={index}>
                            • {warning}
                          </div>
                        )
                      )}

                    </div>

                  </section>
                )}


                <div className="disclaimer">
                  LexTrace AI provides document intelligence and
                  draft assistance. Human verification is required
                  before professional use.
                </div>

              </>
            )}

          </>
        )}


        {/* =======================================================
            DOCUMENTS PAGE
        ======================================================= */}

        {activePage === "Documents" && (
          <>
            <section className="hero">

              <div className="eyebrow">
                Document workspace
              </div>

              <h1>
                Documents <span>Library</span>
              </h1>

              <p>
                Review documents that have been analyzed by the
                LexTrace pipeline.
              </p>

            </section>


            <div className="section-kicker">
              02 · Documents
            </div>

            <h2>
              Processed documents
            </h2>

            <p className="section-copy">
              The current session's analyzed document appears here.
            </p>


            {!documentData && (
              <section className="content-section">

                <div className="empty-state">

                  <div className="empty-state-icon">
                    ◫
                  </div>

                  <h3>
                    No document analyzed yet
                  </h3>

                  <p>
                    Go to Analyze, upload a legal document, and
                    run the LexTrace pipeline.
                  </p>

                  <button
                    type="button"
                    className="analyze-button"
                    onClick={() =>
                      setActivePage("Analyze")
                    }
                  >
                    Analyze a document
                  </button>

                </div>

              </section>
            )}


            {documentData && (
              <section className="content-section">

                <div className="document-card">

                  <div className="document-card-header">

                    <div>

                      <div className="document-card-label">
                        ANALYZED DOCUMENT
                      </div>

                      <h3>
                        {file?.name ||
                          getSourceFilename(
                            documentData.source_path
                          )}
                      </h3>

                    </div>

                    <span className="document-status">
                      Analyzed
                    </span>

                  </div>


                  <div className="document-details">

                    <DocumentDetail
                      label="Document type"
                      value={formatDocType(
                        documentData.doc_type
                      )}
                    />

                    <DocumentDetail
                      label="Source quality"
                      value={`${documentData.confidence_flags?.quality_score ?? 0}/100`}
                    />

                    <DocumentDetail
                      label="Pages"
                      value={
                        documentData.page_count ?? 1
                      }
                    />

                    <DocumentDetail
                      label="Structured fields"
                      value={
                        Object.keys(
                          documentData.structured_fields || {}
                        ).length
                      }
                    />

                  </div>


                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() =>
                      setActivePage("Review")
                    }
                  >
                    Open review →
                  </button>

                </div>

              </section>
            )}

          </>
        )}


        {/* =======================================================
            REVIEW PAGE
        ======================================================= */}

        {activePage === "Review" && (
          <>
            <section className="hero">

              <div className="eyebrow">
                Human-in-the-loop workflow
              </div>

              <h1>
                Human <span>Review</span>
              </h1>

              <p>
                Review extracted facts, source evidence, and
                generated drafts before professional use.
              </p>

            </section>


            <div className="section-kicker">
              03 · Review
            </div>

            <h2>
              Review workspace
            </h2>

            <p className="section-copy">
              Verify the information extracted from the source and
              inspect how the grounded draft was produced.
            </p>


            {!documentData && (
              <section className="content-section">

                <div className="empty-state">

                  <div className="empty-state-icon">
                    ✓
                  </div>

                  <h3>
                    Nothing to review yet
                  </h3>

                  <p>
                    Analyze a document first to populate the review
                    workspace.
                  </p>

                  <button
                    type="button"
                    className="analyze-button"
                    onClick={() =>
                      setActivePage("Analyze")
                    }
                  >
                    Go to Analyze
                  </button>

                </div>

              </section>
            )}


            {documentData && (
              <>

                {/* REVIEW SUMMARY */}

                <div className="metrics">

                  <Metric
                    label="Document"
                    value={formatDocType(
                      documentData.doc_type
                    )}
                    detail="Detected type"
                  />

                  <Metric
                    label="Fields"
                    value={
                      Object.keys(
                        documentData.structured_fields || {}
                      ).length
                    }
                    detail="Extracted fields"
                  />

                  <Metric
                    label="Warnings"
                    value={
                      documentData.warnings?.length ?? 0
                    }
                    detail="Extraction warnings"
                  />

                  <Metric
                    label="Draft sections"
                    value={
                      draftData?.sections?.length ?? 0
                    }
                    detail="Generated sections"
                  />

                </div>


                {/* FIELD REVIEW */}

                <section className="content-section">

                  <div className="section-header">

                    <h3>
                      Extraction review
                    </h3>

                    <span>
                      Verify uncertain fields against source
                      evidence
                    </span>

                  </div>


                  <div className="field-grid">

                    {Object.entries(
                      documentData.structured_fields || {}
                    ).map(([key, value]) => {

                      const evidence =
                        documentData.field_evidence?.[key];

                      const status =
                        evidence?.status || "verified";

                      return (
                        <div
                          className="field-card"
                          key={key}
                        >

                          <div className="field-name">

                            {formatFieldName(key)}

                            <span
                              className={`badge ${status}`}
                            >
                              {status}
                            </span>

                          </div>

                          <div className="field-value">
                            {formatValue(value)}
                          </div>

                          {evidence?.page && (
                            <div className="field-evidence">
                              Source page{" "}
                              {evidence.page}
                            </div>
                          )}

                          {evidence?.excerpt && (
                            <div className="evidence-excerpt">
                              "{evidence.excerpt}"
                            </div>
                          )}

                          {evidence?.reason && (
                            <div className="field-evidence">
                              {evidence.reason}
                            </div>
                          )}

                        </div>
                      );
                    })}

                  </div>

                </section>


                {/* DRAFT REVIEW */}

                {draftData && (
                  <section className="content-section">

                    <div className="section-header">

                      <h3>
                        Draft review
                      </h3>

                      <span className="grounding">
                        Model-reported grounding:{" "}
                        {Math.round(
                          (draftData.overall_grounding_score ??
                            0) * 100
                        )}
                        %
                      </span>

                    </div>


                    <div className="draft">

                      {draftData.sections?.map(
                        (section, index) => (

                          <div
                            className="draft-section"
                            key={index}
                          >

                            <div className="draft-section-header">

                              <h4>
                                {section.section_name}
                              </h4>

                              <span className="section-grounding">
                                {Math.round(
                                  (section.grounding_score ??
                                    0) * 100
                                )}
                                % supported
                              </span>

                            </div>

                            <p>
                              {section.content ||
                                "[Section not generated]"}
                            </p>

                          </div>

                        )
                      )}

                    </div>

                  </section>
                )}


                <div className="disclaimer">
                  LexTrace AI provides document intelligence and
                  draft assistance. Human verification is required
                  before professional use.
                </div>

              </>
            )}

          </>
        )}


        {/* =======================================================
            EVALUATION PAGE
        ======================================================= */}

        {activePage === "Evaluation" && (
          <>
            <section className="hero">

              <div className="eyebrow">
                Quality and evidence assessment
              </div>

              <h1>
                System <span>Evaluation</span>
              </h1>

              <p>
                Inspect extraction quality, grounding information,
                warnings, and document-level signals.
              </p>

            </section>


            <div className="section-kicker">
              04 · Evaluation
            </div>

            <h2>
              Evaluation dashboard
            </h2>

            <p className="section-copy">
              These indicators describe the current analysis and
              should not be interpreted as production accuracy
              benchmarks.
            </p>


            {!documentData && (
              <section className="content-section">

                <div className="empty-state">

                  <div className="empty-state-icon">
                    ◎
                  </div>

                  <h3>
                    No evaluation data yet
                  </h3>

                  <p>
                    Analyze a document to populate the evaluation
                    dashboard.
                  </p>

                  <button
                    type="button"
                    className="analyze-button"
                    onClick={() =>
                      setActivePage("Analyze")
                    }
                  >
                    Analyze a document
                  </button>

                </div>

              </section>
            )}


            {documentData && (
              <>

                <div className="metrics">

                  <Metric
                    label="Source quality"
                    value={`${documentData.confidence_flags?.quality_score ?? 0}/100`}
                    detail="Document extraction quality"
                  />

                  <Metric
                    label="Grounding"
                    value={`${Math.round(
                      (draftData?.overall_grounding_score ??
                        0) * 100
                    )}%`}
                    detail="Model-reported grounding"
                  />

                  <Metric
                    label="Fields"
                    value={
                      Object.keys(
                        documentData.structured_fields || {}
                      ).length
                    }
                    detail="Structured fields extracted"
                  />

                  <Metric
                    label="Warnings"
                    value={
                      documentData.warnings?.length ?? 0
                    }
                    detail="Extraction warnings"
                  />

                </div>


                {/* EXTRACTION QUALITY */}

                <section className="content-section">

                  <div className="section-header">

                    <h3>
                      Extraction quality
                    </h3>

                  </div>

                  <div className="evaluation-grid">

                    <EvaluationRow
                      label="Source quality"
                      value={`${documentData.confidence_flags?.quality_score ?? 0}/100`}
                    />

                    <EvaluationRow
                      label="Pages detected"
                      value={
                        documentData.page_count ?? 1
                      }
                    />

                    <EvaluationRow
                      label="Structured fields"
                      value={
                        Object.keys(
                          documentData.structured_fields || {}
                        ).length
                      }
                    />

                    <EvaluationRow
                      label="Evidence-linked fields"
                      value={
                        Object.keys(
                          documentData.field_evidence || {}
                        ).length
                      }
                    />

                  </div>

                </section>


                {/* GENERATION */}

                {draftData && (
                  <section className="content-section">

                    <div className="section-header">

                      <h3>
                        Generation evaluation
                      </h3>

                    </div>

                    <div className="evaluation-grid">

                      <EvaluationRow
                        label="Generated sections"
                        value={
                          draftData.sections?.length ?? 0
                        }
                      />

                      <EvaluationRow
                        label="Overall grounding"
                        value={`${Math.round(
                          (draftData.overall_grounding_score ??
                            0) * 100
                        )}%`}
                      />

                      <EvaluationRow
                        label="Draft type"
                        value={
                          formatDocType(
                            draftData.draft_type
                          )
                        }
                      />

                    </div>

                  </section>
                )}


                {/* WARNINGS */}

                <section className="content-section">

                  <div className="section-header">

                    <h3>
                      Extraction warnings
                    </h3>

                  </div>

                  {documentData.warnings?.length > 0 ? (

                    <div className="warning-box">

                      {documentData.warnings.map(
                        (warning, index) => (
                          <div key={index}>
                            • {warning}
                          </div>
                        )
                      )}

                    </div>

                  ) : (

                    <div className="success-box">
                      No extraction warnings were reported
                      for this document.
                    </div>

                  )}

                </section>


                <div className="disclaimer">
                  Evaluation indicators shown here are
                  document-level development signals, not
                  validated production accuracy metrics.
                </div>

              </>
            )}

          </>
        )}

      </main>
    </div>
  );
}


/* ===============================================================
   METRIC COMPONENT
================================================================ */

function Metric({
  label,
  value,
  detail,
}) {
  return (
    <div className="metric-card">

      <div className="metric-label">
        {label}
      </div>

      <div className="metric-value">
        {value}
      </div>

      <div className="metric-detail">
        {detail}
      </div>

    </div>
  );
}


/* ===============================================================
   DOCUMENT DETAIL COMPONENT
================================================================ */

function DocumentDetail({
  label,
  value,
}) {
  return (
    <div className="document-detail">

      <div className="document-detail-label">
        {label}
      </div>

      <div className="document-detail-value">
        {value}
      </div>

    </div>
  );
}


/* ===============================================================
   EVALUATION ROW
================================================================ */

function EvaluationRow({
  label,
  value,
}) {
  return (
    <div className="evaluation-row">

      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>

    </div>
  );
}


/* ===============================================================
   HELPERS
================================================================ */

function formatDocType(type) {
  if (!type) {
    return "Unknown";
  }

  return type
    .split("_")
    .map(
      (word) =>
        word.charAt(0).toUpperCase() +
        word.slice(1)
    )
    .join(" ");
}


function formatFieldName(name) {
  return name
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) =>
      char.toUpperCase()
    );
}


function formatValue(value) {
  if (
    value === null ||
    value === undefined
  ) {
    return "—";
  }

  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }

  if (typeof value === "object") {
    return JSON.stringify(
      value,
      null,
      2
    );
  }

  return String(value);
}


function getSourceFilename(sourcePath) {
  if (!sourcePath) {
    return "Unknown document";
  }

  return sourcePath
    .replaceAll("\\", "/")
    .split("/")
    .pop();
}


/* ===============================================================
   EXPORT
================================================================ */

export default App;