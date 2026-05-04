import { useState, useRef } from "react";
import "./ui.css";

export default function UI() {
  const [url, setUrl] = useState("");
  const [topic, setTopic] = useState("");        // Chủ đề
  const [summaryText, setSummaryText] = useState(""); // Nội dung tóm tắt

  const [loadingSummarize, setLoadingSummarize] = useState(false);
  const [loadingUpload, setLoadingUpload] = useState(false);

  const fileInputRef = useRef(null);

  // Reset kết quả
  const resetResult = () => {
    setTopic("");
    setSummaryText("");
  };

  // Tóm tắt từ YouTube
  const handleSummarize = async () => {
    if (!url) return alert("Vui lòng nhập link YouTube");

    try {
      setLoadingSummarize(true);
      resetResult();

      const res = await fetch("http://localhost:8000/api/summary-link", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      const data = await res.json();

      setTopic(data.topic || data.title || "Không xác định chủ đề");
      setSummaryText(data.summary || "Không có nội dung tóm tắt");
    } catch (err) {
      console.error(err);
      alert("Lỗi khi tóm tắt video");
    } finally {
      setLoadingSummarize(false);
    }
  };

  // Upload video
  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoadingUpload(true);
      resetResult();

      const res = await fetch("http://localhost:8000/api/summary-file", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      setTopic(data.topic || data.title || `Video: ${file.name}`);
      setSummaryText(data.summary || "Đang xử lý video...");
    } catch (err) {
      console.error(err);
      alert("Upload video thất bại");
    } finally {
      setLoadingUpload(false);
      e.target.value = null; // reset input file
    }
  };

  return (
    <>
      <section className="hero container">
        <h1>Hệ Thống Tóm Tắt Video Thông Minh</h1>
        <p>
          Khai phá tri thức từ video. Chuyển đổi nội dung video thành bản tóm tắt thông minh.
        </p>

        {/* Input YouTube */}
        <div className="input-box">
          <input
            type="text"
            placeholder="Dán link YouTube tại đây..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <button
            className={`primary-btn ${loadingSummarize ? "disabled" : ""}`}
            onClick={handleSummarize}
            disabled={loadingSummarize}
          >
            {loadingSummarize ? "Đang tóm tắt..." : "Tóm tắt YouTube"}
          </button>
        </div>

        <div className="divider">hoặc</div>

        {/* Upload Video */}
        <div className="upload-section">
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: "none" }}
            accept="video/*"
            onChange={handleFileChange}
          />
          <button
            className={`primary-btn ${loadingUpload ? "disabled" : ""}`}
            onClick={() => fileInputRef.current.click()}
            disabled={loadingUpload}
          >
            {loadingUpload ? "Đang upload và xử lý..." : "Tải video lên"}
          </button>
        </div>
      </section>

      {/* === KẾT QUẢ === */}
      <div className="container result-section">
        {(topic || summaryText) && (
          <div className="summary-box">
            {/* Phần Chủ đề */}
            {topic && (
              <div className="topic">
                <h2>Chủ đề </h2>
                <p className="topic-content">{topic}</p>
              </div>
            )}

            {/* Phần Nội dung tóm tắt */}
            {summaryText && (
              <div className="summary-text">
                <h2>Nội dung tóm tắt</h2>
                <div className="summary-content">
                  {summaryText.split("\n").map((line, index) => (
                    <p key={index}>{line}</p>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Loading khi đang xử lý */}
        {(loadingSummarize || loadingUpload) && (
          <div className="loading-result">
            <div className="spinner"></div>
            <p>Đang xử lý video, vui lòng chờ...</p>
          </div>
        )}
      </div>
    </>
  );
}