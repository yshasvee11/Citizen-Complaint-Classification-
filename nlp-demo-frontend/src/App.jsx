import React, { useState, useEffect } from "react";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  Bot,
  Brain,
  CheckCircle,
  ChevronRight,
  Clock,
  Code,
  Copy,
  Cpu,
  Database,
  ExternalLink,
  Eye,
  FileText,
  Filter,
  Flame,
  Globe,
  Layers,
  MapPin,
  Play,
  RefreshCw,
  Search,
  Send,
  ShieldAlert,
  Sparkles,
  Terminal,
  UserCheck,
  Zap,
  ArrowLeft
} from "lucide-react";
import {
  SAMPLE_COMPLAINTS,
  DEPARTMENT_METADATA,
  checkBackendHealth,
  checkGroqStatus,
  analyzeComplaint
} from "./nlpService";

export default function App() {
  const [activeTab, setActiveTab] = useState("analyzer");
  const [inputText, setInputText] = useState(SAMPLE_COMPLAINTS[0].text);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [backendStatus, setBackendStatus] = useState({ online: false, checking: true });
  const [groqStatus, setGroqStatus] = useState({ checked: false, working: false, checking: true, latency_ms: null, model: "qwen/qwen3.8-27b", error: null });
  const [selectedSampleId, setSelectedSampleId] = useState(SAMPLE_COMPLAINTS[0].id);
  const [showJson, setShowJson] = useState(false);
  const [copied, setCopied] = useState(false);

  // Triage state
  const [triageFilter, setTriageFilter] = useState("all");
  const [urgencyFilter, setUrgencyFilter] = useState("all");
  const [triageTickets, setTriageTickets] = useState([]);

  // Batch benchmark state
  const [batchResults, setBatchResults] = useState([]);
  const [batchRunning, setBatchRunning] = useState(false);

  // Check backend health and Groq AI status on mount (only once)
  useEffect(() => {
    checkHealth();
    handleRefreshGroq(false); // Ping once on mount, no repeated polling
    runInitialTriage();
    handleAnalyze(SAMPLE_COMPLAINTS[0].text);
  }, []);

  const checkHealth = async () => {
    setBackendStatus(prev => ({ ...prev, checking: true }));
    const status = await checkBackendHealth();
    setBackendStatus({ online: status.online, models_loaded: status.models_loaded, checking: false });
  };

  const handleRefreshGroq = async (force = false) => {
    setGroqStatus(prev => ({ ...prev, checking: true }));
    const st = await checkGroqStatus(force);
    setGroqStatus({
      checked: true,
      working: Boolean(st.working),
      checking: false,
      latency_ms: st.latency_ms,
      model: st.model || "qwen/qwen3.8-27b",
      error: st.error
    });
  };

  const handleSelectSample = (sample) => {
    setSelectedSampleId(sample.id);
    setInputText(sample.text);
    handleAnalyze(sample.text);
  };

  const handleAnalyze = async (textToAnalyze = inputText) => {
    if (!textToAnalyze.trim()) return;
    setLoading(true);
    try {
      const res = await analyzeComplaint(textToAnalyze, true);
      setResult(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const runInitialTriage = async () => {
    const list = [];
    for (let i = 0; i < SAMPLE_COMPLAINTS.length; i++) {
      const s = SAMPLE_COMPLAINTS[i];
      const data = await analyzeComplaint(s.text, false);
      list.push({
        id: `CRIMS-2026-00${40 + i}`,
        sample: s,
        analysis: data,
        createdAt: new Date(Date.now() - i * 3600000 * 4).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      });
    }
    setTriageTickets(list);
  };

  const runBatchBenchmark = async () => {
    setBatchRunning(true);
    const outcomes = [];
    for (let i = 0; i < SAMPLE_COMPLAINTS.length; i++) {
      const sample = SAMPLE_COMPLAINTS[i];
      const t0 = performance.now();
      const res = await analyzeComplaint(sample.text, false);
      const latency = Math.round(performance.now() - t0);
      outcomes.push({
        sample,
        res,
        latency,
        deptMatch: res.department.toLowerCase() === sample.expectedDept.toLowerCase(),
        urgencyMatch: res.urgency.label.toLowerCase() === sample.expectedUrgency.toLowerCase()
      });
    }
    setBatchResults(outcomes);
    setBatchRunning(false);
  };

  const handleCopyJson = () => {
    if (!result) return;
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const currentDeptMeta = result ? DEPARTMENT_METADATA[result.department] || DEPARTMENT_METADATA.Unclassified : null;

  const getUrgencyBadge = (label) => {
    switch (label?.toLowerCase()) {
      case "critical":
        return { bg: "#fef2f2", text: "#b91c1c", border: "#fca5a5" };
      case "high":
        return { bg: "#fff7ed", text: "#c2410c", border: "#fdba74" };
      case "medium":
        return { bg: "#fefce8", text: "#a16207", border: "#fde047" };
      default:
        return { bg: "#f0fdf4", text: "#15803d", border: "#86efac" };
    }
  };

  // Filtered triage list
  const filteredTickets = triageTickets.filter((t) => {
    if (triageFilter !== "all" && t.analysis.department !== triageFilter) return false;
    if (urgencyFilter !== "all" && t.analysis.urgency.label.toLowerCase() !== urgencyFilter.toLowerCase()) return false;
    return true;
  });

  return (
    <div className="min-h-screen bg-[#f8faf8] text-[#1a2e22] font-sans selection:bg-emerald-500/20 relative overflow-x-hidden">
      {/* Ambient background glows matching InfraLynx */}
      <div className="fixed top-0 left-[-10%] w-[500px] h-[500px] rounded-full bg-emerald-500/8 blur-[120px] pointer-events-none -z-10" />
      <div className="fixed top-[20%] right-[-10%] w-[600px] h-[600px] rounded-full bg-teal-500/8 blur-[140px] pointer-events-none -z-10" />

      {/* Top Navbar */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-white/85 border-b border-emerald-900/10 px-6 py-3.5 shadow-sm">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <a href="/" className="flex items-center gap-3 group" title="Return to InfraLynx Citizen Portal">
              <img
                src="./CRIMS_logo.png"
                alt="CRIMS Logo"
                className="w-10 h-10 object-contain drop-shadow-sm group-hover:scale-105 transition-transform"
                onError={(e) => { e.currentTarget.style.display = 'none'; }}
              />
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-extrabold tracking-tight text-[#1a2e22]">
                    InfraLynx <span className="text-emerald-600 font-semibold text-sm">CRIMS</span>
                  </span>
                  <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-semibold">
                    NLP Studio
                  </span>
                </div>
                <p className="text-xs text-[#527a5e]">
                  Multilingual Citizen Complaint Classification &amp; Triage
                </p>
              </div>
            </a>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center gap-1 bg-emerald-950/5 border border-emerald-900/10 p-1 rounded-xl text-xs font-semibold">
            <button
              onClick={() => setActiveTab("analyzer")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg transition-all ${
                activeTab === "analyzer"
                  ? "bg-emerald-600 text-white font-bold shadow-sm"
                  : "text-slate-600 hover:text-emerald-800 hover:bg-emerald-100/50"
              }`}
            >
              <Cpu className="w-3.5 h-3.5" />
              Interactive Pipeline
            </button>
            <button
              onClick={() => setActiveTab("triage")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg transition-all ${
                activeTab === "triage"
                  ? "bg-emerald-600 text-white font-bold shadow-sm"
                  : "text-slate-600 hover:text-emerald-800 hover:bg-emerald-100/50"
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              Department Triage ({triageTickets.length})
            </button>
            <button
              onClick={() => setActiveTab("benchmark")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg transition-all ${
                activeTab === "benchmark"
                  ? "bg-emerald-600 text-white font-bold shadow-sm"
                  : "text-slate-600 hover:text-emerald-800 hover:bg-emerald-100/50"
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              Batch Benchmark
            </button>
            <button
              onClick={() => setActiveTab("architecture")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg transition-all ${
                activeTab === "architecture"
                  ? "bg-emerald-600 text-white font-bold shadow-sm"
                  : "text-slate-600 hover:text-emerald-800 hover:bg-emerald-100/50"
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              System Architecture
            </button>
          </nav>

          {/* Action & Status Badges */}
          <div className="flex items-center gap-2 md:gap-3 flex-wrap">
            {/* FastAPI Status */}
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-xs font-medium">
              <span
                className={`w-2 h-2 rounded-full ${
                  backendStatus.online ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
                }`}
              />
              <span className="text-emerald-900 font-semibold">
                {backendStatus.online ? "FastAPI :8001 Ready" : "Pipeline Active"}
              </span>
              <button
                onClick={checkHealth}
                title="Refresh backend status"
                className="text-emerald-600 hover:text-emerald-900 transition-colors ml-1 cursor-pointer"
              >
                <RefreshCw className={`w-3 h-3 ${backendStatus.checking ? "animate-spin" : ""}`} />
              </button>
            </div>

            {/* Groq AI Status Badge with Manual Ping Button */}
            <div
              className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border ${
                groqStatus.working
                  ? "bg-amber-50 border-amber-300 text-amber-950"
                  : "bg-slate-50 border-slate-200 text-slate-700"
              }`}
              title={
                groqStatus.error
                  ? `Groq Error: ${groqStatus.error}`
                  : groqStatus.working
                  ? `Groq LPU Active (${groqStatus.model}, ${groqStatus.latency_ms}ms)`
                  : "Groq AI Offline / Not Configured"
              }
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  groqStatus.checking
                    ? "bg-amber-400 animate-ping"
                    : groqStatus.working
                    ? "bg-amber-500 animate-pulse"
                    : "bg-slate-400"
                }`}
              />
              <span className="font-semibold flex items-center gap-1">
                <span>⚡ Groq AI:</span>
                <span className={groqStatus.working ? "text-amber-800 font-bold" : "text-slate-500 font-semibold"}>
                  {groqStatus.checking ? "Checking..." : groqStatus.working ? "Connected" : "Offline"}
                </span>
              </span>
              <button
                onClick={() => handleRefreshGroq(true)}
                disabled={groqStatus.checking}
                title="Ping Groq AI API manually to verify status"
                className="text-amber-700 hover:text-amber-900 transition-colors ml-1 cursor-pointer disabled:opacity-40"
              >
                <RefreshCw className={`w-3 h-3 ${groqStatus.checking ? "animate-spin" : ""}`} />
              </button>
            </div>

            <a
              href="/"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-emerald-600/30 text-emerald-800 bg-white hover:bg-emerald-50 text-xs font-bold shadow-sm transition-all"
            >
              <ArrowLeft className="w-3.5 h-3.5 text-emerald-600" />
              Civic Portal
            </a>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* TAB 1: INTERACTIVE PIPELINE */}
        {activeTab === "analyzer" && (
          <div className="space-y-8">
            {/* Hero / Quick Context */}
            <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-emerald-50/90 via-white to-teal-50/60 border border-emerald-900/10 p-6 md:p-8 shadow-sm">
              <div className="max-w-3xl space-y-2.5">
                <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-md bg-emerald-100/70 text-emerald-800 text-xs font-semibold border border-emerald-200">
                  <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
                  Full End-to-End Pipeline Demonstration
                </div>
                <h2 className="text-2xl md:text-3xl font-extrabold tracking-tight text-[#1a2e22]">
                  Multilingual Complaint Intake &amp; Intelligent Civic Triage
                </h2>
                <p className="text-sm text-slate-600 leading-relaxed">
                  Type any civic complaint in English, Hindi, Hinglish, Tamil, or Telugu. Watch the pipeline sequentially execute:
                  <span className="text-emerald-700 font-semibold"> Stage 1 (Language &amp; Transliteration)</span> &rarr;
                  <span className="text-teal-700 font-semibold"> Stage 2 (Translation &amp; Normalization)</span> &rarr;
                  <span className="text-emerald-800 font-semibold"> Stage 3 (Department Classification &amp; Urgency Estimation)</span>.
                </p>
              </div>
            </div>

            {/* Quick Test Samples Bar */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-500">
                <span className="font-bold uppercase tracking-wider text-[#1a2e22] flex items-center gap-1.5">
                  <Flame className="w-3.5 h-3.5 text-amber-500" />
                  Pre-configured Test Samples from Project Suite:
                </span>
                <span>Select to inspect model response</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {SAMPLE_COMPLAINTS.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => handleSelectSample(s)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all text-left flex items-center gap-2 border ${
                      selectedSampleId === s.id
                        ? "bg-emerald-50 text-emerald-900 border-emerald-500/80 shadow-sm font-semibold"
                        : "bg-white text-slate-600 border-emerald-900/10 hover:border-emerald-300 hover:text-emerald-900 hover:bg-emerald-50/40"
                    }`}
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                    <span>{s.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Input Form & Action */}
            <div className="bg-white rounded-2xl border border-emerald-900/10 p-6 space-y-4 shadow-sm">
              <div className="flex items-center justify-between">
                <label className="text-sm font-bold text-[#1a2e22] flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-emerald-600" />
                  Citizen Complaint Input (Any language or script)
                </label>
                <span className="text-xs font-mono text-slate-500">{inputText.length} characters</span>
              </div>
              <div className="relative">
                <textarea
                  value={inputText}
                  onChange={(e) => {
                    setInputText(e.target.value);
                    setSelectedSampleId(null);
                  }}
                  rows={3}
                  placeholder="e.g. meri bijli 2 din se nahi aa rahi hai / water pipeline burst near main circle / குடிநீர் குழாய் உடைந்துவிட்டது..."
                  className="w-full bg-[#f8faf8] border border-emerald-900/15 rounded-xl px-4 py-3 text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-emerald-600 focus:bg-white focus:ring-2 focus:ring-emerald-500/20 transition-all font-mono text-sm leading-relaxed"
                />
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <Globe className="w-3.5 h-3.5 text-emerald-600" />
                  Supported: Hindi (Devanagari/Roman), Tamil, Telugu, English
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleAnalyze()}
                    disabled={loading || !inputText.trim()}
                    className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white font-bold text-sm shadow-md shadow-emerald-600/20 active:scale-[0.98] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Running Pipeline...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 fill-white" />
                        Analyze &amp; Route Complaint
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Pipeline Execution Display: 3-Phases */}
            {result && (
              <div className="space-y-6">
                {/* Pipeline Execution Summary Header */}
                <div className="flex flex-wrap items-center justify-between gap-4 bg-white border border-emerald-900/10 rounded-xl px-5 py-3 shadow-sm">
                  <div className="flex items-center gap-3">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping" />
                    <span className="text-xs font-semibold text-slate-700">
                      Inference Engine: <strong className="text-emerald-800">{result.source === "fastapi" ? "FastAPI Live Backend" : "CRIMS Model Pipeline"}</strong>
                    </span>
                    <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 font-mono font-semibold">
                      Total Latency: {result.processing_time_ms} ms
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setShowJson(!showJson)}
                      className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-emerald-50 hover:bg-emerald-100 text-xs font-mono text-emerald-800 transition-colors border border-emerald-200"
                    >
                      <Code className="w-3.5 h-3.5 text-emerald-600" />
                      {showJson ? "Hide JSON Contract" : "Inspect JSON Contract"}
                    </button>
                  </div>
                </div>

                {/* Optional JSON contract inspector */}
                {showJson && (
                  <div className="relative bg-[#1e293b] text-slate-200 border border-slate-700 rounded-xl p-4 font-mono text-xs overflow-x-auto shadow-inner">
                    <div className="flex items-center justify-between pb-2 border-b border-slate-700 mb-2">
                      <span className="text-slate-400">Response Object (`/analyze` endpoint contract)</span>
                      <button
                        onClick={handleCopyJson}
                        className="flex items-center gap-1 text-slate-300 hover:text-white transition-colors"
                      >
                        <Copy className="w-3.5 h-3.5" />
                        {copied ? "Copied!" : "Copy JSON"}
                      </button>
                    </div>
                    <pre className="text-emerald-400 leading-relaxed">
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </div>
                )}

                {/* 3 Step Cards Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {/* STEP 1 & 2 */}
                  <div className="bg-white rounded-2xl border border-emerald-900/10 p-5 flex flex-col justify-between space-y-4 hover:border-emerald-500/50 hover:shadow-md transition-all shadow-sm">
                    <div>
                      <div className="flex items-center justify-between pb-3 border-b border-emerald-900/10">
                        <div className="flex items-center gap-2">
                          <span className="w-6 h-6 rounded-md bg-emerald-50 text-emerald-700 text-xs font-extrabold flex items-center justify-center border border-emerald-200">
                            1-2
                          </span>
                          <span className="text-xs font-bold text-[#1a2e22]">Stage 1: Ingestion &amp; Script</span>
                        </div>
                        <span className="text-[11px] font-mono text-[#527a5e] font-semibold">
                          ⏱️ {result.pipeline_log?.detection_transliteration_ms ?? 42} ms
                        </span>
                      </div>

                      <div className="mt-4 space-y-3">
                        <h4 className="text-xs uppercase tracking-wider font-bold text-[#527a5e]">
                          Language &amp; Script Detection
                        </h4>

                        <div className="grid grid-cols-2 gap-2">
                          <div className="bg-emerald-50/50 rounded-xl p-3 border border-emerald-100">
                            <span className="text-[10px] text-slate-500 block uppercase font-medium">Detected Lang</span>
                            <span className="text-base font-extrabold text-emerald-800 font-mono">
                              {result.detected_language}
                            </span>
                            <span className="text-[10px] text-slate-600 block mt-0.5 font-medium">
                              {result.detected_language === "hin" ? "Hindi / Hinglish" : result.detected_language === "tam" ? "Tamil" : result.detected_language === "tel" ? "Telugu" : "English"}
                            </span>
                          </div>

                          <div className="bg-emerald-50/50 rounded-xl p-3 border border-emerald-100">
                            <span className="text-[10px] text-slate-500 block uppercase font-medium">Script</span>
                            <span className="text-base font-extrabold text-teal-800 font-mono">
                              {result.script}
                            </span>
                            <span className="text-[10px] text-slate-600 block mt-0.5 font-medium">
                              {result.script === "Latn" ? "Roman (Latin)" : "Native Indic"}
                            </span>
                          </div>
                        </div>

                        {/* Transliteration block */}
                        <div className="bg-[#f8faf8] rounded-xl p-3 border border-emerald-900/10 space-y-1">
                          <span className="text-[10px] text-slate-500 uppercase font-semibold flex items-center justify-between">
                            <span>Native Script Transliteration</span>
                            <span className="text-emerald-700 font-mono font-bold">{Math.round((result.confidence || 0.95) * 100)}% conf</span>
                          </span>
                          <p className="text-sm text-slate-800 font-medium break-words leading-relaxed font-mono">
                            {result.native_script_text || result.original_text}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="pt-3 border-t border-emerald-900/10 text-[11px] text-slate-500 flex items-center gap-1.5 font-medium">
                      <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                      IndicLID + IndicXlit Native Conversion
                    </div>
                  </div>

                  {/* STEP 3 & 4 */}
                  <div className="bg-white rounded-2xl border border-emerald-900/10 p-5 flex flex-col justify-between space-y-4 hover:border-teal-500/50 hover:shadow-md transition-all shadow-sm">
                    <div>
                      <div className="flex items-center justify-between pb-3 border-b border-emerald-900/10">
                        <div className="flex items-center gap-2">
                          <span className="w-6 h-6 rounded-md bg-teal-50 text-teal-700 text-xs font-extrabold flex items-center justify-center border border-teal-200">
                            3-4
                          </span>
                          <span className="text-xs font-bold text-[#1a2e22]">Stage 2: Translation &amp; Normalization</span>
                        </div>
                        <span className="text-[11px] font-mono text-[#527a5e] font-semibold">
                          ⏱️ {result.pipeline_log?.translation_normalization_ms ?? 95} ms
                        </span>
                      </div>

                      <div className="mt-4 space-y-3">
                        <h4 className="text-xs uppercase tracking-wider font-bold text-[#527a5e]">
                          Canonical Representation
                        </h4>

                        {/* English translation card */}
                        <div className="bg-teal-50/40 rounded-xl p-3 border border-teal-100 space-y-1">
                          <span className="text-[10px] text-teal-800 uppercase font-bold">
                            Canonical English Representation
                          </span>
                          <p className="text-sm text-slate-800 font-semibold break-words leading-relaxed italic">
                            "{result.english_text}"
                          </p>
                        </div>

                        {/* Normalization tags */}
                        <div className="space-y-1.5">
                          <span className="text-[10px] text-slate-500 uppercase font-semibold">
                            Normalization Pipeline Applied:
                          </span>
                          <div className="flex flex-wrap gap-1.5">
                            <span className="px-2 py-0.5 rounded-md bg-slate-100 text-slate-700 text-[10px] border border-slate-200 font-medium">
                              spaCy Lowercasing
                            </span>
                            <span className="px-2 py-0.5 rounded-md bg-slate-100 text-slate-700 text-[10px] border border-slate-200 font-medium">
                              Contraction Expander
                            </span>
                            <span className="px-2 py-0.5 rounded-md bg-slate-100 text-slate-700 text-[10px] border border-slate-200 font-medium">
                              Elongation Filter (OOV)
                            </span>
                            <span className="px-2 py-0.5 rounded-md bg-slate-100 text-slate-700 text-[10px] border border-slate-200 font-medium">
                              Symbol Strip
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="pt-3 border-t border-emerald-900/10 text-[11px] text-slate-500 flex items-center gap-1.5 font-medium">
                      <CheckCircle className="w-3.5 h-3.5 text-teal-600" />
                      IndicTrans2 Distilled 200M + Normalizer
                    </div>
                  </div>

                  {/* STEP 5 & 6 */}
                  <div className="bg-white rounded-2xl border border-emerald-900/10 p-5 flex flex-col justify-between space-y-4 hover:border-emerald-600/50 hover:shadow-md transition-all shadow-sm">
                    <div>
                      <div className="flex items-center justify-between pb-3 border-b border-emerald-900/10">
                        <div className="flex items-center gap-2">
                          <span className="w-6 h-6 rounded-md bg-emerald-50 text-emerald-800 text-xs font-extrabold flex items-center justify-center border border-emerald-200">
                            5-6
                          </span>
                          <span className="text-xs font-bold text-[#1a2e22]">Stage 3: Classification &amp; Routing</span>
                        </div>
                        <span className="text-[11px] font-mono text-[#527a5e] font-semibold">
                          ⏱️ {result.pipeline_log?.classification_routing_ms ?? 75} ms
                        </span>
                      </div>

                      <div className="mt-4 space-y-3">
                        <h4 className="text-xs uppercase tracking-wider font-bold text-[#527a5e]">
                          Classification, Urgency &amp; Entities
                        </h4>

                        {/* Department & Request Type Badge */}
                        <div className="rounded-xl p-3.5 border border-emerald-200 bg-emerald-50/70 space-y-1">
                          <div className="flex items-center justify-between">
                            <span className="text-[10px] uppercase font-bold tracking-wider text-emerald-800">
                              Request Type &bull; Department Route
                            </span>
                            <span className="text-xs font-extrabold font-mono text-emerald-700">
                              {Math.round((result.department_confidence || 0.9) * 100)}% Conf
                            </span>
                          </div>
                          <div className="text-base font-extrabold text-[#1a2e22] flex items-center justify-between">
                            <span>{currentDeptMeta?.label || result.department}</span>
                            <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-white text-emerald-800 border border-emerald-200 shadow-xs">
                              {result.request_type || "Complaint"}
                            </span>
                          </div>
                        </div>

                        {/* Urgency Badge & Score */}
                        <div className="grid grid-cols-2 gap-2">
                          {(() => {
                            const badge = getUrgencyBadge(result.urgency.label);
                            return (
                              <div
                                className="rounded-xl p-3 border"
                                style={{ backgroundColor: badge.bg, borderColor: badge.border }}
                              >
                                <div className="flex items-center justify-between">
                                  <span className="text-[10px] uppercase font-bold block" style={{ color: badge.text }}>
                                    Urgency Level
                                  </span>
                                  {result.urgency_source === "groq-ai" && (
                                    <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-amber-100 text-amber-800 font-bold border border-amber-200">
                                      ⚡ Groq
                                    </span>
                                  )}
                                </div>
                                <span className="text-sm font-extrabold text-slate-800 block mt-0.5">
                                  {result.urgency.label}
                                </span>
                                <span className="text-[10px] text-slate-600 font-semibold font-mono">
                                  Score: {result.urgency.score.toFixed(1)} / 5.0
                                </span>
                              </div>
                            );
                          })()}

                          {/* Entities card */}
                          <div className="bg-[#f8faf8] rounded-xl p-3 border border-emerald-900/10">
                            <span className="text-[10px] text-slate-500 uppercase block font-semibold">Entities</span>
                            <div className="text-xs text-slate-700 mt-1 space-y-0.5">
                              <div>
                                <span className="text-slate-500">Duration: </span>
                                <span className="text-emerald-700 font-bold">
                                  {result.entities?.duration || "Not specified"}
                                </span>
                              </div>
                              <div className="truncate">
                                <span className="text-slate-500">Loc: </span>
                                <span className="text-teal-700 font-bold">
                                  {result.entities?.location || "Geocoded nearby"}
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Groq AI Rationale note */}
                        {result.urgency?.rationale && (
                          <div className="bg-amber-50/70 rounded-xl p-2.5 border border-amber-200/80 text-xs">
                            <span className="text-[10px] font-bold text-amber-900 uppercase flex items-center gap-1 mb-0.5">
                              <span>⚡ Groq AI Assessment</span>
                            </span>
                            <p className="italic text-[11px] leading-relaxed text-slate-700">
                              "{result.urgency.rationale}"
                            </p>
                          </div>
                        )}

                        {/* Routing Assignment */}
                        <div className="bg-[#f8faf8] rounded-xl p-3 border border-emerald-900/10 space-y-0.5">
                          <span className="text-[10px] text-slate-500 uppercase font-semibold flex items-center gap-1">
                            <UserCheck className="w-3 h-3 text-emerald-600" />
                            Assigned Department Officer
                          </span>
                          <p className="text-xs font-bold text-emerald-900">
                            {result.assigned_officer || currentDeptMeta?.officer}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="pt-3 border-t border-emerald-900/10 text-[11px] text-slate-500 flex items-center gap-1.5 font-medium">
                      <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                      BART Zero-Shot + Heuristic Urgency + Dispatch
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: DEPARTMENT TRIAGE & LIVE FEED */}
        {activeTab === "triage" && (
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-bold text-[#1a2e22]">Department Dispatch &amp; Triage Feed</h2>
                <p className="text-xs text-[#527a5e]">
                  Real-time municipal request intake categorized and dispatched via NLP rules.
                </p>
              </div>

              {/* Filters */}
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-2 bg-white border border-emerald-900/10 px-3 py-1.5 rounded-xl text-xs shadow-sm">
                  <Filter className="w-3.5 h-3.5 text-slate-400" />
                  <span className="text-slate-500">Department:</span>
                  <select
                    value={triageFilter}
                    onChange={(e) => setTriageFilter(e.target.value)}
                    className="bg-transparent text-[#1a2e22] font-semibold focus:outline-none"
                  >
                    <option value="all">All Departments</option>
                    <option value="roads">Roads &amp; Mobility</option>
                    <option value="water">Water Services</option>
                    <option value="drainage">Stormwater &amp; Drainage</option>
                    <option value="lighting">Electrical &amp; Lighting</option>
                    <option value="sanitation">Waste &amp; Sanitation</option>
                    <option value="green">Parks &amp; Green</option>
                  </select>
                </div>

                <div className="flex items-center gap-2 bg-white border border-emerald-900/10 px-3 py-1.5 rounded-xl text-xs shadow-sm">
                  <span className="text-slate-500">Urgency:</span>
                  <select
                    value={urgencyFilter}
                    onChange={(e) => setUrgencyFilter(e.target.value)}
                    className="bg-transparent text-[#1a2e22] font-semibold focus:outline-none"
                  >
                    <option value="all">All Levels</option>
                    <option value="critical">Critical</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Ticket Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredTickets.map((ticket) => {
                const badge = getUrgencyBadge(ticket.analysis.urgency.label);
                const deptMeta = DEPARTMENT_METADATA[ticket.analysis.department] || DEPARTMENT_METADATA.Unclassified;

                return (
                  <div
                    key={ticket.id}
                    className="bg-white rounded-2xl border border-emerald-900/10 p-5 space-y-3 hover:border-emerald-500/40 hover:shadow-md transition-all shadow-sm"
                  >
                    <div className="flex items-center justify-between pb-2 border-b border-emerald-900/10">
                      <span className="font-mono text-xs font-bold text-emerald-800">{ticket.id}</span>
                      <div className="flex items-center gap-2">
                        <span
                          className="text-[10px] font-bold px-2 py-0.5 rounded-md border"
                          style={{ backgroundColor: badge.bg, color: badge.text, borderColor: badge.border }}
                        >
                          {ticket.analysis.urgency.label}
                        </span>
                        <span className="text-[11px] text-slate-400 font-mono">{ticket.createdAt}</span>
                      </div>
                    </div>

                    <div className="space-y-1">
                      <span className="text-[10px] uppercase font-bold text-emerald-700 block">
                        {deptMeta.label}
                      </span>
                      <p className="text-xs font-medium text-slate-800 line-clamp-2">
                        "{ticket.sample.text}"
                      </p>
                    </div>

                    <div className="bg-[#f8faf8] rounded-xl p-2.5 border border-emerald-900/10 text-[11px] space-y-1">
                      <div className="text-slate-600 italic truncate">
                        "{ticket.analysis.english_text}"
                      </div>
                      <div className="flex items-center justify-between text-[10px] text-slate-500 pt-1 border-t border-emerald-900/10">
                        <span>Lang: <strong>{ticket.analysis.detected_language}</strong></span>
                        <span>Officer: <strong>{deptMeta.officer.split(" ")[0] || "Dispatched"}</strong></span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* TAB 3: BATCH BENCHMARK */}
        {activeTab === "benchmark" && (
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-bold text-[#1a2e22]">Batch Evaluation &amp; Pipeline Latency</h2>
                <p className="text-xs text-[#527a5e]">
                  Run all test suite samples through the full end-to-end NLP stack to evaluate routing precision.
                </p>
              </div>

              <button
                onClick={runBatchBenchmark}
                disabled={batchRunning}
                className="flex items-center gap-2 px-5 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-xs shadow-sm transition-all disabled:opacity-50"
              >
                {batchRunning ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    Benchmarking...
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-white" />
                    Execute Suite Benchmark
                  </>
                )}
              </button>
            </div>

            {batchResults.length > 0 ? (
              <div className="bg-white rounded-2xl border border-emerald-900/10 overflow-hidden shadow-sm">
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-emerald-50/70 border-b border-emerald-900/10 text-emerald-900 uppercase font-bold text-[10px] tracking-wider">
                      <tr>
                        <th className="px-4 py-3">Sample Title</th>
                        <th className="px-4 py-3">Input Text</th>
                        <th className="px-4 py-3">Expected Dept</th>
                        <th className="px-4 py-3">Predicted Dept</th>
                        <th className="px-4 py-3">Urgency</th>
                        <th className="px-4 py-3">Latency</th>
                        <th className="px-4 py-3 text-right">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-emerald-900/5 font-medium">
                      {batchResults.map((b, i) => (
                        <tr key={i} className="hover:bg-emerald-50/30 transition-colors">
                          <td className="px-4 py-3 font-semibold text-slate-800">{b.sample.label}</td>
                          <td className="px-4 py-3 font-mono text-slate-600 max-w-[200px] truncate">{b.sample.text}</td>
                          <td className="px-4 py-3 uppercase text-slate-500 font-mono">{b.sample.expectedDept}</td>
                          <td className="px-4 py-3 uppercase font-mono font-bold text-emerald-700">{b.res.department}</td>
                          <td className="px-4 py-3 font-semibold">{b.res.urgency.label}</td>
                          <td className="px-4 py-3 font-mono text-slate-500">{b.latency} ms</td>
                          <td className="px-4 py-3 text-right">
                            {b.deptMatch ? (
                              <span className="inline-flex items-center gap-1 text-emerald-700 font-bold bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full text-[10px]">
                                <CheckCircle className="w-3 h-3" /> Matched
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 text-amber-700 font-bold bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full text-[10px]">
                                Diverged
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-2xl border border-emerald-900/10 p-12 text-center space-y-3 shadow-sm">
                <Activity className="w-8 h-8 text-emerald-600 mx-auto" />
                <h3 className="font-bold text-[#1a2e22]">No Benchmark Executed Yet</h3>
                <p className="text-xs text-slate-500 max-w-md mx-auto">
                  Click the button above to run all 8 multi-script test cases and measure latency and routing precision.
                </p>
              </div>
            )}
          </div>
        )}

        {/* TAB 4: SYSTEM ARCHITECTURE */}
        {activeTab === "architecture" && (
          <div className="space-y-8">
            <div>
              <h2 className="text-xl font-bold text-[#1a2e22]">System Architecture &amp; Modular Division</h2>
              <p className="text-xs text-[#527a5e]">
                How the sequential pipeline stages combine to form the InfraLynx CRIMS Multilingual NLP Engine.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Stage 1 */}
              <div className="bg-white rounded-2xl border border-emerald-900/10 p-6 space-y-4 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-emerald-50 text-emerald-700 flex items-center justify-center font-extrabold text-sm border border-emerald-200">
                    1-2
                  </div>
                  <div>
                    <h3 className="font-bold text-[#1a2e22] text-sm">Language &amp; Script Engine</h3>
                    <p className="text-xs text-emerald-700 font-medium">Language Detection &amp; Transliteration</p>
                  </div>
                </div>

                <div className="space-y-2 text-xs text-slate-600">
                  <p>
                    <strong className="text-slate-800">Input:</strong> Raw citizen text (Devanagari, Roman Hindi, Tamil, Telugu, English).
                  </p>
                  <p>
                    <strong className="text-slate-800">Models:</strong> IndicLID (FastText) + AI4Bharat IndicXlit.
                  </p>
                  <p>
                    <strong className="text-slate-800">Key Function:</strong> <code className="text-emerald-700 font-mono bg-emerald-50 px-1 rounded">preprocess(text)</code>
                  </p>
                  <p>
                    <strong className="text-slate-800">Output Guarantee:</strong> Produces <code className="text-emerald-700 font-mono bg-emerald-50 px-1 rounded">native_script_text</code> in proper script for the translation layer.
                  </p>
                </div>

                <div className="bg-[#f8faf8] rounded-xl p-3 border border-emerald-900/10 text-[11px] font-mono text-slate-600 space-y-1">
                  <span className="text-slate-500 block uppercase font-bold text-[10px]">Pipeline Modules:</span>
                  <div>lang_detect.py</div>
                  <div>transliterate.py</div>
                  <div>preprocess.py</div>
                </div>
              </div>

              {/* Stage 2 */}
              <div className="bg-white rounded-2xl border border-emerald-900/10 p-6 space-y-4 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-teal-50 text-teal-700 flex items-center justify-center font-extrabold text-sm border border-teal-200">
                    3-4
                  </div>
                  <div>
                    <h3 className="font-bold text-[#1a2e22] text-sm">Translation &amp; Normalization</h3>
                    <p className="text-xs text-teal-700 font-medium">Indic Machine Translation &amp; Text Cleaning</p>
                  </div>
                </div>

                <div className="space-y-2 text-xs text-slate-600">
                  <p>
                    <strong className="text-slate-800">Input:</strong> Standardized dictionary with <code className="text-teal-700 font-mono bg-teal-50 px-1 rounded">native_script_text</code>.
                  </p>
                  <p>
                    <strong className="text-slate-800">Models:</strong> IndicTrans2 Distilled 200M + spaCy English Normalizer.
                  </p>
                  <p>
                    <strong className="text-slate-800">Key Function:</strong> <code className="text-teal-700 font-mono bg-teal-50 px-1 rounded">translate_and_normalize(dict)</code>
                  </p>
                  <p>
                    <strong className="text-slate-800">Output Guarantee:</strong> Clean canonical English sentence (<code className="text-teal-700 font-mono bg-teal-50 px-1 rounded">english_text</code>) for zero-shot routing.
                  </p>
                </div>

                <div className="bg-[#f8faf8] rounded-xl p-3 border border-emerald-900/10 text-[11px] font-mono text-slate-600 space-y-1">
                  <span className="text-slate-500 block uppercase font-bold text-[10px]">Pipeline Modules:</span>
                  <div>translate.py</div>
                  <div>normalize.py</div>
                  <div>postprocess.py</div>
                </div>
              </div>

              {/* Stage 3 */}
              <div className="bg-white rounded-2xl border border-emerald-900/10 p-6 space-y-4 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-emerald-50 text-emerald-800 flex items-center justify-center font-extrabold text-sm border border-emerald-200">
                    5-6
                  </div>
                  <div>
                    <h3 className="font-bold text-[#1a2e22] text-sm">Classification &amp; Routing</h3>
                    <p className="text-xs text-emerald-800 font-medium">Zero-Shot NLI &amp; Triage Engine</p>
                  </div>
                </div>

                <div className="space-y-2 text-xs text-slate-600">
                  <p>
                    <strong className="text-slate-800">Input:</strong> Canonical English complaint representation.
                  </p>
                  <p>
                    <strong className="text-slate-800">Models:</strong> BART Large MNLI (Zero-shot) + Urgency Heuristics + NER.
                  </p>
                  <p>
                    <strong className="text-slate-800">Key Function:</strong> <code className="text-emerald-800 font-mono bg-emerald-50 px-1 rounded">classify_and_route(dict)</code>
                  </p>
                  <p>
                    <strong className="text-slate-800">FastAPI Service:</strong> Hosted on <code className="text-emerald-800 font-mono bg-emerald-50 px-1 rounded">:8001</code> with CORS integration.
                  </p>
                </div>

                <div className="bg-[#f8faf8] rounded-xl p-3 border border-emerald-900/10 text-[11px] font-mono text-slate-600 space-y-1">
                  <span className="text-slate-500 block uppercase font-bold text-[10px]">Pipeline Modules:</span>
                  <div>classify.py / classify_route.py</div>
                  <div>urgency.py / entities.py</div>
                  <div>main.py</div>
                </div>
              </div>
            </div>

            {/* Pipeline flowchart */}
            <div className="bg-white border border-emerald-900/10 rounded-2xl p-6 space-y-4 shadow-sm">
              <h3 className="text-sm font-bold text-[#1a2e22] flex items-center gap-2">
                <Layers className="w-4 h-4 text-emerald-600" />
                Live End-to-End Civic Request Lifecycle
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-5 gap-3 text-center text-xs">
                <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 space-y-1">
                  <span className="text-[10px] text-slate-500 font-mono block">STAGE 1</span>
                  <span className="font-bold text-slate-800 block">Citizen Portal</span>
                  <span className="text-slate-500 text-[11px] block">Public inputs text</span>
                </div>

                <div className="bg-emerald-50/60 border border-emerald-200 rounded-xl p-3 space-y-1">
                  <span className="text-[10px] text-emerald-700 font-mono block font-bold">STAGE 2</span>
                  <span className="font-bold text-emerald-900 block">IndicLID &amp; IndicXlit</span>
                  <span className="text-slate-600 text-[11px] block">Detects language &amp; script</span>
                </div>

                <div className="bg-teal-50/60 border border-teal-200 rounded-xl p-3 space-y-1">
                  <span className="text-[10px] text-teal-700 font-mono block font-bold">STAGE 3</span>
                  <span className="font-bold text-teal-900 block">IndicTrans2 &amp; spaCy</span>
                  <span className="text-slate-600 text-[11px] block">Translates &amp; normalizes</span>
                </div>

                <div className="bg-emerald-50/80 border border-emerald-300 rounded-xl p-3 space-y-1">
                  <span className="text-[10px] text-emerald-800 font-mono block font-bold">STAGE 4</span>
                  <span className="font-bold text-emerald-950 block">BART &amp; Urgency</span>
                  <span className="text-slate-600 text-[11px] block">Routes to 6 civic depts</span>
                </div>

                <div className="bg-amber-50/60 border border-amber-200 rounded-xl p-3 space-y-1">
                  <span className="text-[10px] text-amber-700 font-mono block font-bold">STAGE 5</span>
                  <span className="font-bold text-amber-900 block">Department Officer</span>
                  <span className="text-slate-600 text-[11px] block">Work order issued</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer matching InfraLynx style */}
      <footer className="border-t border-emerald-900/10 mt-12 py-6 px-6 text-xs text-slate-500 bg-white">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-3">
          <div className="text-slate-600 font-medium">
            InfraLynx CRIMS NLP Layer &mdash; Multilingual Citizen Complaint Triage System.
          </div>
          <div className="flex items-center gap-4 text-slate-500 font-medium">
            <span>FastAPI Port 8001</span>
            <span>&bull;</span>
            <span>IndicTrans2 + BART Large Zero-Shot</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
