import { useState, useEffect, useRef, useCallback } from "react";
import Icon from "@/components/ui/icon";
import { analyzeFrame, computeFlickerStats, resetAnalyzerState, EVENT_COOLDOWN_MS } from "@/lib/screenAnalyzer";
import { predict, getTopPatterns } from "@/lib/mlPredictor";
import { aiPredict, getThoughtLog, resetAI, saveMemory, loadMemory, clearMemory, hasSavedMemory, getSavedMemoryMeta } from "@/lib/neuralPredictor";
import type { RoundResult, FlickerSample, FrameAnalysis } from "@/lib/screenAnalyzer";
import type { Prediction, Pattern } from "@/lib/mlPredictor";
import type { AIPrediction, AIThought } from "@/lib/neuralPredictor";

type Tab = "capture" | "model";

function fmt(ts: number) {
  return new Date(ts).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function ReactorBadge({ reactor, size = "md" }: { reactor: "alpha" | "omega" | null; size?: "sm" | "md" | "lg" }) {
  if (!reactor) return <span className="text-white/30 font-mono">—</span>;
  const sizes = { sm: "text-xs px-2 py-0.5", md: "text-sm px-3 py-1", lg: "text-xl px-5 py-2 font-display tracking-widest" };
  const styles = reactor === "alpha"
    ? "bg-cyan-500/15 border border-cyan-500/40 text-cyan-300"
    : "bg-purple-500/15 border border-purple-500/40 text-purple-300";
  return (
    <span className={`rounded-lg font-bold ${sizes[size]} ${styles}`}>
      {reactor === "alpha" ? "α Альфа" : "ω Омега"}
    </span>
  );
}

function ConfBar({ value, color = "#00ffcc" }: { value: number; color?: string }) {
  return (
    <div className="h-2 rounded-full bg-white/5 overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-500"
        style={{ width: `${value * 100}%`, background: color, boxShadow: `0 0 8px ${color}88` }}
      />
    </div>
  );
}

export default function Index() {
  const [tab, setTab] = useState<Tab>("capture");
  const [step, setStep] = useState<"idle" | "preview" | "analyzing">("idle");
  const [capturing, setCapturing] = useState(false);
  const [history, setHistory] = useState<RoundResult[]>([]);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [aiPrediction, setAiPrediction] = useState<AIPrediction | null>(null);
  const [aiThoughts, setAiThoughts] = useState<AIThought[]>([]);
  const [memoryStatus, setMemoryStatus] = useState<"none" | "saved" | "loaded">("none");
  const [memoryMeta, setMemoryMeta] = useState<{ rounds: number; hypothesesCount: number; savedAt: number } | null>(null);
  const [lastFrame, setLastFrame] = useState<FrameAnalysis | null>(null);
  const [flickerSamples, setFlickerSamples] = useState<FlickerSample[]>([]);
  const [roundPhase, setRoundPhase] = useState<"idle" | "flicker" | "result">("idle");
  const [countdown, setCountdown] = useState<number>(30);
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [liveTime, setLiveTime] = useState(new Date());
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [selectionMode, setSelectionMode] = useState(false);
  const [cropRect, setCropRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [drawCurrent, setDrawCurrent] = useState<{ x: number; y: number } | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analyzeLoopRef = useRef<number | null>(null);
  const roundIdRef = useRef(1);
  const lastResultTsRef = useRef<number>(0);
  const flickerBufRef = useRef<FlickerSample[]>([]);
  const phaseRef = useRef<"idle" | "flicker" | "result">("idle");

  // Живые часы
  useEffect(() => {
    const t = setInterval(() => setLiveTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // Проверяем наличие сохранённой памяти при старте
  useEffect(() => {
    if (hasSavedMemory()) {
      setMemoryMeta(getSavedMemoryMeta());
      setMemoryStatus("saved");
    }
  }, []);

  // Обратный отсчёт 30 секунд
  useEffect(() => {
    if (!capturing) return;
    const t = setInterval(() => {
      setCountdown(prev => {
        const next = prev <= 1 ? 30 : prev - 1;
        return next;
      });
    }, 1000);
    return () => clearInterval(t);
  }, [capturing]);

  const stopCapture = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (analyzeLoopRef.current) cancelAnimationFrame(analyzeLoopRef.current);
    setCapturing(false);
    setStep("idle");
    setCropRect(null);
    setSelectionMode(false);
    phaseRef.current = "idle";
    setRoundPhase("idle");
    flickerBufRef.current = [];
    setAiPrediction(null);
    setAiThoughts([]);
    resetAI();
  }, []);

  const handleSaveMemory = useCallback(() => {
    saveMemory();
    const meta = getSavedMemoryMeta();
    setMemoryMeta(meta);
    setMemoryStatus("saved");
  }, []);

  const handleLoadMemory = useCallback(() => {
    const result = loadMemory();
    if (result.ok) {
      setMemoryMeta(getSavedMemoryMeta());
      setMemoryStatus("loaded");
      setAiThoughts([...getThoughtLog().slice(-10)]);
    }
  }, []);

  const handleClearMemory = useCallback(() => {
    clearMemory();
    setMemoryStatus("none");
    setMemoryMeta(null);
    setAiPrediction(null);
    setAiThoughts([]);
  }, []);

  // Шаг 1: запустить стрим и показать превью
  const startPreview = useCallback(async () => {
    setCaptureError(null);
    setCaptureError("[1] Проверяю API...");

    if (!navigator.mediaDevices?.getDisplayMedia) {
      setCaptureError(`[СТОП] getDisplayMedia недоступен\nisSecureContext: ${window.isSecureContext}\nprotocol: ${location.protocol}`);
      return;
    }

    setCaptureError("[2] Вызываю getDisplayMedia...");

    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
      const tracks = stream.getVideoTracks();
      setCaptureError(`[3] Стрим получен ✓\nТреки: ${tracks.map(t => t.label || t.kind).join(", ")}\nvideoRef сейчас: ${videoRef.current ? "есть" : "null — жду рендера"}`);

      streamRef.current = stream;
      stream.getVideoTracks()[0].addEventListener("ended", stopCapture);

      // Сначала рендерим video-элемент (step=preview), потом в useEffect присвоим srcObject
      setStep("preview");
      setSelectionMode(true);
    } catch (e: unknown) {
      const err = e as Error;
      setCaptureError(`[ОШИБКА] ${err.name}: ${err.message}`);
    }
  }, [stopCapture]);

  // Шаг 3: начать анализ после выделения области
  const startAnalyzing = useCallback(() => {
    resetAnalyzerState();
    lastResultTsRef.current = 0;
    flickerBufRef.current = [];
    setCapturing(true);
    setStep("analyzing");
    setSelectionMode(false);
  }, []);

  // Присваиваем srcObject после рендера video-элемента
  useEffect(() => {
    if (step === "preview" && videoRef.current && streamRef.current) {
      const video = videoRef.current;
      video.srcObject = streamRef.current;
      setCaptureError(prev => (prev ?? "") + `\n[4] srcObject присвоен ✓\nvideo.readyState: ${video.readyState}`);
      video.onloadedmetadata = () => {
        setCaptureError(prev => (prev ?? "") + `\n[5] metadata загружена ✓ — ${video.videoWidth}×${video.videoHeight}`);
        video.play().then(() => {
          setCaptureError(prev => (prev ?? "") + "\n[6] play() ✓ — превью должно отображаться");
        }).catch(e => {
          setCaptureError(prev => (prev ?? "") + `\n[6] play() ОШИБКА: ${e.message}`);
        });
      };
    }
  }, [step]);

  // Основной цикл анализа кадров
  useEffect(() => {
    if (!capturing) return;

    const loop = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) {
        analyzeLoopRef.current = requestAnimationFrame(loop);
        return;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 360;

      // Если задан cropRect — рисуем только выделенную область
      if (cropRect) {
        const previewEl = previewRef.current;
        const displayW = previewEl?.clientWidth || vw;
        const displayH = previewEl?.clientHeight || vh;
        const scaleX = vw / displayW;
        const scaleY = vh / displayH;
        const sx = cropRect.x * scaleX;
        const sy = cropRect.y * scaleY;
        const sw = cropRect.w * scaleX;
        const sh = cropRect.h * scaleY;
        canvas.width = Math.max(sw, 2);
        canvas.height = Math.max(sh, 2);
        ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
      } else {
        canvas.width = vw;
        canvas.height = vh;
        ctx.drawImage(video, 0, 0, vw, vh);
      }

      const frame = analyzeFrame(ctx, canvas.width, canvas.height);
      setLastFrame(frame);

      const now = frame.timestamp;
      const prevPhase = phaseRef.current;

      // Мерцание — собираем сэмплы с учётом смены доминирующей стороны
      if (frame.phase === "flicker" || frame.phase === "result") {
        if (phaseRef.current === "idle") {
          phaseRef.current = "flicker";
          setRoundPhase("flicker");
        }

        // Фиксируем смену доминирующей стороны
        const prevBuf = flickerBufRef.current;
        const lastDom = prevBuf.length > 0 ? prevBuf[prevBuf.length - 1].dominant : null;
        const switchEvent = frame.dominant !== null && lastDom !== null && frame.dominant !== lastDom;

        if (frame.dominant !== null) {
          flickerBufRef.current.push({
            timestamp: now,
            dominant: frame.dominant,
            alphaLevel: frame.alphaSmooth,
            omegaLevel: frame.omegaSmooth,
            switchEvent,
          });
        }

        // Держим окно 8 секунд для полного анализа мерцания
        flickerBufRef.current = flickerBufRef.current.filter(s => now - s.timestamp < 8000);
        setFlickerSamples([...flickerBufRef.current]);
      }

      // Резкий скачок = победитель зафиксирован
      if (frame.phase === "result" && frame.winner && now - lastResultTsRef.current > EVENT_COOLDOWN_MS) {
        lastResultTsRef.current = now;
        phaseRef.current = "result";
        setRoundPhase("result");

        const flickerStats = computeFlickerStats(flickerBufRef.current);

        setHistory(prev => {
          const prevPred = predict(prev, flickerStats.bias, flickerStats.rate, flickerStats.switchCount);
          const predictedBefore = prevPred.reactor;
          const predictionHit = predictedBefore !== null ? predictedBefore === frame.winner : null;

          const newResult: RoundResult = {
            id: roundIdRef.current++,
            winner: frame.winner,
            timestamp: now,
            flickerPattern: [...flickerBufRef.current],
            flickerRate: flickerStats.rate,
            flickerSwitchCount: flickerStats.switchCount,
            flickerBias: flickerStats.bias,
            lastFlickerDominant: flickerStats.lastDominant,
            predictedBefore,
            predictionHit,
          };

          const next = [...prev, newResult];
          const nextPred = predict(next, flickerStats.bias, flickerStats.rate, flickerStats.switchCount);
          setPrediction(nextPred);
          setPatterns(getTopPatterns(next));

          // ИИ: обучаем на реальном победителе, затем строим следующий прогноз
          const aiResult = aiPredict(next, flickerStats.bias, flickerStats.rate, flickerStats.switchCount, frame.winner);
          setAiPrediction(aiResult);
          setAiThoughts([...getThoughtLog().slice(-10)]);

          return next;
        });

        flickerBufRef.current = [];
        setFlickerSamples([]);

        setTimeout(() => {
          phaseRef.current = "idle";
          setRoundPhase("idle");
          resetAnalyzerState();
        }, 3000);
      }

      // Сигнал пропал — возвращаемся в idle
      if (frame.phase === "idle" && phaseRef.current === "flicker") {
        phaseRef.current = "idle";
        setRoundPhase("idle");
      }

      analyzeLoopRef.current = requestAnimationFrame(loop);
    };

    analyzeLoopRef.current = requestAnimationFrame(loop);
    return () => {
      if (analyzeLoopRef.current) cancelAnimationFrame(analyzeLoopRef.current);
    };
  }, [capturing]);

  const flickerStats = computeFlickerStats(flickerSamples);
  const totalRounds = history.length;
  const alphaWins = history.filter(r => r.winner === "alpha").length;
  const omegaWins = history.filter(r => r.winner === "omega").length;

  const TABS: { id: Tab; icon: string; label: string }[] = [
    { id: "capture", icon: "Monitor", label: "Захват" },
    { id: "model", icon: "Cpu", label: "Модель ML" },
  ];

  const phaseColor = roundPhase === "result" ? "#00ffcc" : roundPhase === "flicker" ? "#facc15" : "#64748b";
  const phaseLabel = roundPhase === "result" ? "РЕЗУЛЬТАТ" : roundPhase === "flicker" ? "МЕРЦАНИЕ" : "ОЖИДАНИЕ";

  return (
    <div className="min-h-screen pattern-dot">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-neon-green/10 bg-[#080d14]/90 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-neon-green/10 border border-neon-green/30 flex items-center justify-center">
                <Icon name="Atom" size={16} className="text-neon-green" />
              </div>
              <span className="font-display text-white text-lg tracking-widest">REACTOR<span className="text-neon-green">OS</span></span>
            </div>
            <div className="flex items-center gap-3">
              <span className="font-mono text-xs text-white/30 hidden md:inline">{liveTime.toLocaleTimeString("ru-RU")}</span>
              <div
                className="flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-mono font-medium transition-all duration-300"
                style={{ borderColor: `${phaseColor}44`, background: `${phaseColor}11`, color: phaseColor }}
              >
                <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: phaseColor }} />
                {phaseLabel}
              </div>
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-neon-purple/10 border border-neon-purple/20">
                <Icon name="Database" size={12} className="text-neon-purple" />
                <span className="text-neon-purple text-xs font-mono">{totalRounds} раундов</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Tabs */}
      <div className="sticky top-14 z-40 bg-[#080d14]/80 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="flex overflow-x-auto">
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`flex items-center gap-2 px-4 sm:px-5 py-4 text-xs font-display tracking-widest transition-all duration-300 whitespace-nowrap ${
                  tab === t.id ? "tab-active" : "tab-inactive hover:text-white/70"
                }`}
              >
                <Icon name={t.icon} size={13} />
                <span className="hidden sm:inline">{t.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Всегда в DOM — не размонтируются при смене вкладки */}
      <video ref={videoRef} className="hidden" autoPlay muted playsInline />
      <canvas ref={canvasRef} className="hidden" />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-5">

        {/* ── ЗАХВАТ ЭКРАНА ── */}
        {tab === "capture" && (
          <div className="animate-fade-in-up">
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-5 items-start">

            {/* ── ЛЕВАЯ КОЛОНКА ── */}
            <div className="space-y-5">

            {/* Шаги */}
            <div className="glass-card rounded-xl p-5">

              {/* Индикатор шагов */}
              <div className="flex items-center gap-2 mb-5">
                {[
                  { n: 1, label: "Захват окна", done: step !== "idle" },
                  { n: 2, label: "Выделить область", done: !!cropRect },
                  { n: 3, label: "Начать анализ", done: step === "analyzing" },
                ].map((s, i) => (
                  <div key={s.n} className="flex items-center gap-2">
                    <div className="flex items-center gap-1.5">
                      <div
                        className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-display transition-all"
                        style={{
                          background: s.done ? "rgba(0,255,204,0.2)" : step !== "idle" && s.n === (step === "preview" ? 2 : 3) ? "rgba(250,204,21,0.2)" : "rgba(255,255,255,0.05)",
                          border: s.done ? "1px solid rgba(0,255,204,0.5)" : step !== "idle" && s.n === (step === "preview" ? 2 : 3) ? "1px solid rgba(250,204,21,0.5)" : "1px solid rgba(255,255,255,0.1)",
                          color: s.done ? "#00ffcc" : step !== "idle" && s.n === (step === "preview" ? 2 : 3) ? "#facc15" : "rgba(255,255,255,0.3)",
                        }}
                      >{s.done ? "✓" : s.n}</div>
                      <span className="text-xs font-mono hidden sm:inline" style={{ color: s.done ? "#00ffcc" : "rgba(255,255,255,0.3)" }}>{s.label}</span>
                    </div>
                    {i < 2 && <div className="w-6 h-px bg-white/10" />}
                  </div>
                ))}
              </div>

              <div className="flex flex-wrap gap-3 items-center">
                {/* Шаг 1 */}
                {step === "idle" && (
                  <button
                    onClick={startPreview}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-display tracking-widest text-sm transition-all"
                    style={{ background: "linear-gradient(135deg, #00ffcc, #38bdf8)", color: "#080d14", boxShadow: "0 0 20px rgba(0,255,204,0.3)" }}
                  >
                    <Icon name="Monitor" size={15} />
                    Шаг 1 — Захватить окно игры
                  </button>
                )}

                {/* Шаг 2 */}
                {step === "preview" && (
                  <>
                    {!cropRect ? (
                      <div className="px-4 py-2.5 rounded-xl border border-yellow-400/30 bg-yellow-400/8">
                        <p className="font-mono text-xs text-yellow-400">↓ Нарисуй прямоугольник мышкой на превью ниже</p>
                      </div>
                    ) : (
                      <button
                        onClick={startAnalyzing}
                        className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-display tracking-widest text-sm transition-all"
                        style={{ background: "linear-gradient(135deg, #00ffcc, #a855f7)", color: "#080d14", boxShadow: "0 0 20px rgba(0,255,204,0.3)" }}
                      >
                        <Icon name="Play" size={15} />
                        Шаг 3 — Начать анализ
                      </button>
                    )}
                    <button
                      onClick={() => { setSelectionMode(true); setCropRect(null); }}
                      className="flex items-center gap-1.5 px-3 py-2.5 rounded-xl text-xs font-display tracking-widest border transition-all"
                      style={{ borderColor: "rgba(250,204,21,0.4)", color: "#facc15", background: "rgba(250,204,21,0.08)" }}
                    >
                      <Icon name="Crop" size={12} />
                      {cropRect ? "Перевыделить" : "Выделить область"}
                    </button>
                  </>
                )}

                {/* Шаг 3 — идёт анализ */}
                {step === "analyzing" && (
                  <>
                    <span className="flex items-center gap-2 text-sm font-display text-neon-green">
                      <span className="w-2 h-2 rounded-full bg-neon-green animate-pulse" style={{ boxShadow: "0 0 8px #00ffcc" }} />
                      Анализ идёт...
                    </span>
                    <button
                      onClick={() => { setStep("preview"); setCapturing(false); setCropRect(null); setSelectionMode(true); }}
                      className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-display tracking-widest border border-yellow-400/30 text-yellow-400 hover:bg-yellow-400/10 transition-all"
                    >
                      <Icon name="Crop" size={12} />
                      Изменить область
                    </button>
                  </>
                )}

                {/* Стоп — всегда если не idle */}
                {step !== "idle" && (
                  <button
                    onClick={stopCapture}
                    className="flex items-center gap-2 px-4 py-2.5 rounded-xl font-display tracking-widest text-xs border border-red-500/40 text-red-400 hover:bg-red-500/10 transition-all"
                  >
                    <Icon name="Square" size={13} />
                    Остановить
                  </button>
                )}
              </div>

              {captureError && (
                <div className="mt-4 px-4 py-3 rounded-lg bg-white/5 border border-white/10 font-mono text-xs leading-relaxed whitespace-pre-wrap"
                  style={{ color: captureError.includes("ОШИБКА") || captureError.includes("СТОП") ? "#f87171" : "#a3e635" }}>
                  {captureError}
                </div>
              )}
            </div>

            {/* Превью захвата — показываем с шага preview */}
            {step !== "idle" && (
            <div className="glass-card rounded-xl overflow-hidden">
              <div className="px-5 pt-4 pb-2 flex items-center justify-between flex-wrap gap-2">
                <span className="font-display text-xs tracking-widest text-white/40 uppercase">Превью захваченного экрана</span>
                <div className="flex items-center gap-2">
                  {step === "analyzing" && (
                    <span className="flex items-center gap-1.5 text-xs font-mono text-red-400">
                      <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
                      REC
                    </span>
                  )}
                  {step === "preview" && selectionMode && (
                    <span className="font-mono text-xs text-yellow-400 animate-pulse">✏️ Режим выделения</span>
                  )}
                </div>
              </div>

              {selectionMode && (
                <div className="mx-4 mb-2 px-3 py-2 rounded-lg bg-yellow-400/10 border border-yellow-400/30">
                  <p className="font-mono text-xs text-yellow-400">Нарисуй прямоугольник мышкой поверх двух реакторов (Альфа слева, Омега справа)</p>
                </div>
              )}

              {cropRect && !selectionMode && (
                <div className="mx-4 mb-2 px-3 py-2 rounded-lg bg-neon-green/10 border border-neon-green/20 flex items-center gap-2">
                  <Icon name="CheckCircle" size={12} className="text-neon-green" />
                  <p className="font-mono text-xs text-neon-green">
                    Область задана: {Math.round(cropRect.x)},{Math.round(cropRect.y)} → {Math.round(cropRect.w)}×{Math.round(cropRect.h)}px
                  </p>
                </div>
              )}

              <div
                ref={previewRef}
                className="relative bg-black/40 mx-4 mb-4 rounded-lg overflow-hidden"
                style={{ height: "calc(100vh - 320px)", minHeight: 360, cursor: selectionMode ? "crosshair" : "default" }}
                onMouseDown={e => {
                  if (!selectionMode) return;
                  const rect = e.currentTarget.getBoundingClientRect();
                  const x = e.clientX - rect.left;
                  const y = e.clientY - rect.top;
                  setDrawing(true);
                  setDrawStart({ x, y });
                  setDrawCurrent({ x, y });
                }}
                onMouseMove={e => {
                  if (!selectionMode || !drawing) return;
                  const rect = e.currentTarget.getBoundingClientRect();
                  setDrawCurrent({ x: e.clientX - rect.left, y: e.clientY - rect.top });
                }}
                onMouseUp={e => {
                  if (!selectionMode || !drawing || !drawStart) return;
                  const rect = e.currentTarget.getBoundingClientRect();
                  const ex = e.clientX - rect.left;
                  const ey = e.clientY - rect.top;
                  const x = Math.min(drawStart.x, ex);
                  const y = Math.min(drawStart.y, ey);
                  const w = Math.abs(ex - drawStart.x);
                  const h = Math.abs(ey - drawStart.y);
                  if (w > 10 && h > 10) {
                    setCropRect({ x, y, w, h });
                    setSelectionMode(false);
                  }
                  setDrawing(false);
                  setDrawStart(null);
                  setDrawCurrent(null);
                }}
              >
                {step !== "idle" ? (
                  <>
                    <video
                      className="w-full h-full object-contain"
                      autoPlay muted playsInline
                      ref={el => { if (el && streamRef.current && el.srcObject !== streamRef.current) el.srcObject = streamRef.current; }}
                    />

                    {/* Оверлей выделения */}
                    {selectionMode && drawing && drawStart && drawCurrent && (
                      <div
                        className="absolute border-2 border-yellow-400 pointer-events-none"
                        style={{
                          left: Math.min(drawStart.x, drawCurrent.x),
                          top: Math.min(drawStart.y, drawCurrent.y),
                          width: Math.abs(drawCurrent.x - drawStart.x),
                          height: Math.abs(drawCurrent.y - drawStart.y),
                          background: "rgba(250,204,21,0.08)",
                          boxShadow: "0 0 0 1px rgba(250,204,21,0.3)",
                        }}
                      >
                        <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-full px-2 py-0.5 bg-yellow-400 text-black text-xs font-mono whitespace-nowrap" style={{ fontSize: 10 }}>
                          {Math.round(Math.abs(drawCurrent.x - drawStart.x))} × {Math.round(Math.abs(drawCurrent.y - drawStart.y))}
                        </div>
                      </div>
                    )}

                    {/* Активная область анализа */}
                    {cropRect && !selectionMode && (
                      <div
                        className="absolute pointer-events-none"
                        style={{
                          left: cropRect.x, top: cropRect.y,
                          width: cropRect.w, height: cropRect.h,
                          border: "2px solid rgba(0,255,204,0.6)",
                          boxShadow: "0 0 12px rgba(0,255,204,0.2), inset 0 0 12px rgba(0,255,204,0.04)",
                        }}
                      >
                        <div className="absolute top-1 left-1 text-xs font-display text-cyan-400/80 tracking-widest" style={{ fontSize: 10 }}>α</div>
                        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-white/20" />
                        <div className="absolute top-1 right-1 text-xs font-display text-purple-400/80 tracking-widest" style={{ fontSize: 10 }}>ω</div>
                      </div>
                    )}

                    {/* Подсказка если нет выделения */}
                    {!cropRect && !selectionMode && (
                      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                        <div className="px-4 py-2 rounded-lg bg-black/60 border border-yellow-400/30">
                          <p className="font-mono text-xs text-yellow-400/80">Нажми «Выделить область реакторов»</p>
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="flex flex-col items-center justify-center h-60 gap-3">
                    <Icon name="Monitor" size={40} className="text-white/10" />
                    <p className="font-mono text-xs text-white/20">Захват не активен</p>
                  </div>
                )}
              </div>
            </div>
            )}

            {/* Живой анализ пикселей */}
            {capturing && lastFrame && (
              <div className="grid grid-cols-2 gap-4">
                {/* Альфа */}
                <div className="glass-card rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="w-3 h-3 rounded-full bg-cyan-400" style={{ boxShadow: "0 0 8px #22d3ee" }} />
                    <span className="font-display text-xs tracking-widest text-cyan-400">α РЕАКТОР АЛЬФА</span>
                  </div>
                  <div className="mb-2">
                    <div className="flex justify-between mb-1">
                      <span className="font-mono text-xs text-white/40">Жёлтый сигнал</span>
                      <span className="font-mono text-xs text-yellow-400">{(lastFrame.alphaYellow * 100).toFixed(2)}%</span>
                    </div>
                    <ConfBar value={Math.min(lastFrame.alphaYellow * 10, 1)} color="#facc15" />
                  </div>
                  <div className="mb-2">
                    <div className="flex justify-between mb-1">
                      <span className="font-mono text-xs text-white/40">Скачок (Δ)</span>
                      <span className="font-mono text-xs" style={{ color: lastFrame.alphaDelta >= 0.015 ? "#00ffcc" : lastFrame.alphaDelta > 0 ? "#facc15" : "rgba(255,255,255,0.3)" }}>
                        {lastFrame.alphaDelta >= 0 ? "+" : ""}{(lastFrame.alphaDelta * 100).toFixed(2)}%
                        {lastFrame.alphaDelta >= 0.015 && " ⚡"}
                      </span>
                    </div>
                    <ConfBar value={Math.min(Math.abs(lastFrame.alphaDelta) * 20, 1)} color={lastFrame.alphaDelta >= 0.015 ? "#00ffcc" : "#facc1555"} />
                  </div>
                  {roundPhase === "result" && lastFrame.winner === "alpha" && (
                    <div className="mt-3 text-center py-2 rounded-lg bg-neon-green/10 border border-neon-green/30">
                      <span className="font-display text-sm text-neon-green tracking-widest">✓ ЗАФИКСИРОВАНО</span>
                    </div>
                  )}
                </div>

                {/* Омега */}
                <div className="glass-card-purple rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="w-3 h-3 rounded-full bg-purple-400" style={{ boxShadow: "0 0 8px #c084fc" }} />
                    <span className="font-display text-xs tracking-widest text-purple-400">ω РЕАКТОР ОМЕГА</span>
                  </div>
                  <div className="mb-2">
                    <div className="flex justify-between mb-1">
                      <span className="font-mono text-xs text-white/40">Жёлтый сигнал</span>
                      <span className="font-mono text-xs text-yellow-400">{(lastFrame.omegaYellow * 100).toFixed(2)}%</span>
                    </div>
                    <ConfBar value={Math.min(lastFrame.omegaYellow * 10, 1)} color="#facc15" />
                  </div>
                  <div className="mb-2">
                    <div className="flex justify-between mb-1">
                      <span className="font-mono text-xs text-white/40">Скачок (Δ)</span>
                      <span className="font-mono text-xs" style={{ color: lastFrame.omegaDelta >= 0.015 ? "#c084fc" : lastFrame.omegaDelta > 0 ? "#facc15" : "rgba(255,255,255,0.3)" }}>
                        {lastFrame.omegaDelta >= 0 ? "+" : ""}{(lastFrame.omegaDelta * 100).toFixed(2)}%
                        {lastFrame.omegaDelta >= 0.015 && " ⚡"}
                      </span>
                    </div>
                    <ConfBar value={Math.min(Math.abs(lastFrame.omegaDelta) * 20, 1)} color={lastFrame.omegaDelta >= 0.015 ? "#c084fc" : "#facc1555"} />
                  </div>
                  {roundPhase === "result" && lastFrame.winner === "omega" && (
                    <div className="mt-3 text-center py-2 rounded-lg bg-purple-500/10 border border-purple-500/30">
                      <span className="font-display text-sm text-purple-300 tracking-widest">✓ ЗАФИКСИРОВАНО</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Мерцание в реальном времени */}
            {capturing && (
              <div className="glass-card rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Icon name="Zap" size={13} className="text-yellow-400" />
                    <span className="font-display text-xs tracking-widest text-yellow-400 uppercase">Мерцание (live)</span>
                  </div>
                  {/* Кто мерцает сейчас */}
                  {flickerStats.lastDominant && (
                    <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs font-mono"
                      style={{
                        background: flickerStats.lastDominant === "alpha" ? "rgba(34,211,238,0.15)" : "rgba(192,132,252,0.15)",
                        border: `1px solid ${flickerStats.lastDominant === "alpha" ? "rgba(34,211,238,0.4)" : "rgba(192,132,252,0.4)"}`,
                        color: flickerStats.lastDominant === "alpha" ? "#22d3ee" : "#c084fc",
                      }}
                    >
                      <span className="animate-pulse">●</span>
                      {flickerStats.lastDominant === "alpha" ? "α Альфа" : "ω Омега"}
                    </div>
                  )}
                </div>

                {/* Метрики */}
                <div className="grid grid-cols-4 gap-2 mb-3">
                  <div className="text-center">
                    <p className="font-mono text-xs text-white/30 mb-0.5">Темп</p>
                    <p className="font-display text-lg text-yellow-400">{flickerStats.rate.toFixed(1)}<span className="text-xs text-white/30">/с</span></p>
                  </div>
                  <div className="text-center">
                    <p className="font-mono text-xs text-white/30 mb-0.5">α↔ω / сек</p>
                    <p className="font-display text-lg text-white/80">
                      {flickerStats.rate.toFixed(2)}
                      <span className="text-xs text-white/30 ml-0.5">({flickerStats.switchCount})</span>
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="font-mono text-xs text-white/30 mb-0.5">α %</p>
                    <p className="font-display text-lg text-cyan-400">{Math.round(flickerStats.alphaPct * 100)}%</p>
                  </div>
                  <div className="text-center">
                    <p className="font-mono text-xs text-white/30 mb-0.5">ω %</p>
                    <p className="font-display text-lg text-purple-400">{Math.round(flickerStats.omegaPct * 100)}%</p>
                  </div>
                </div>

                {/* Шкала частоты переключений */}
                <div className="mb-2">
                  <div className="flex justify-between mb-1">
                    <span className="font-mono text-white/25" style={{ fontSize: 9 }}>редко</span>
                    <span className="font-mono text-white/40" style={{ fontSize: 10 }}>
                      частота α↔ω: {flickerStats.rate < 0.5 ? "нет сигнала" : flickerStats.rate < 1.5 ? "медленно" : flickerStats.rate < 3 ? "умеренно" : flickerStats.rate < 5 ? "быстро" : "очень быстро"}
                    </span>
                    <span className="font-mono text-white/25" style={{ fontSize: 9 }}>часто</span>
                  </div>
                  <div className="h-2 rounded-full bg-white/5 overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-300"
                      style={{
                        width: `${Math.min(flickerStats.rate / 6 * 100, 100)}%`,
                        background: flickerStats.rate < 1 ? "#475569"
                          : flickerStats.rate < 2 ? "#facc15"
                          : flickerStats.rate < 4 ? "#fb923c"
                          : "#f43f5e",
                      }}
                    />
                  </div>
                </div>

                {/* Визуализация смен — каждый столбик = кадр, высота = уровень сигнала */}
                <div className="flex gap-px items-end" style={{ height: 36 }}>
                  {flickerSamples.slice(-50).map((s, i, arr) => {
                    const level = Math.max(s.alphaLevel, s.omegaLevel);
                    const h = Math.max(4, Math.min(36, level * 800));
                    const isSwitch = s.switchEvent;
                    return (
                      <div
                        key={i}
                        className="flex-1 rounded-sm transition-all"
                        style={{
                          height: h,
                          background: s.dominant === "alpha" ? "rgba(34,211,238,0.8)" : "rgba(192,132,252,0.8)",
                          boxShadow: isSwitch ? `0 0 6px ${s.dominant === "alpha" ? "#22d3ee" : "#c084fc"}` : "none",
                          opacity: 0.4 + (i / arr.length) * 0.6,
                          outline: isSwitch ? `1px solid ${s.dominant === "alpha" ? "#22d3ee" : "#c084fc"}` : "none",
                        }}
                      />
                    );
                  })}
                </div>
                <div className="flex justify-between mt-1">
                  <span className="font-mono text-xs text-white/15">← 8 сек назад</span>
                  <span className="font-mono text-xs text-yellow-400/60">сейчас →</span>
                </div>
              </div>
            )}

            {/* Предсказание */}
            {prediction && prediction.reactor && (
              <div className="rounded-xl border overflow-hidden"
                style={{
                  background: prediction.reactor === "alpha" ? "rgba(34,211,238,0.05)" : "rgba(192,132,252,0.05)",
                  borderColor: prediction.reactor === "alpha" ? "rgba(34,211,238,0.3)" : "rgba(192,132,252,0.3)",
                  boxShadow: prediction.reactor === "alpha" ? "0 0 24px rgba(34,211,238,0.08)" : "0 0 24px rgba(192,132,252,0.08)",
                }}
              >
                {/* Основной результат */}
                <div className="flex items-center justify-between px-5 py-4">
                  <div>
                    <p className="font-mono text-xs text-white/30 uppercase tracking-widest mb-2">Следующий победитель</p>
                    <ReactorBadge reactor={prediction.reactor} size="lg" />
                    <p className="font-mono text-xs text-white/35 mt-2 leading-relaxed max-w-xs">{prediction.reason}</p>
                  </div>
                  <div className="text-center flex-shrink-0">
                    <p className="font-display text-5xl" style={{ color: prediction.reactor === "alpha" ? "#22d3ee" : "#c084fc", textShadow: `0 0 24px currentColor` }}>
                      {Math.round(prediction.confidence * 100)}%
                    </p>
                    <p className="font-mono text-xs text-white/30 mt-1">уверенность</p>
                  </div>
                </div>

                {/* Разбивка сигналов */}
                <div className="border-t px-5 py-3 grid grid-cols-4 gap-x-3 gap-y-2" style={{ borderColor: "rgba(255,255,255,0.06)" }}>
                  {[
                    { label: "Паттерн", val: prediction.signals.patternScore, color: "#00ffcc" },
                    { label: "Мерц.↔Пат", val: prediction.signals.flickerPatternScore, color: "#34d399" },
                    { label: "Мерцание", val: prediction.signals.flickerScore, color: "#facc15" },
                    { label: "Баланс", val: prediction.signals.balanceScore, color: "#38bdf8" },
                    { label: "Серия 6+", val: prediction.signals.streakScore, color: "#a855f7" },
                    { label: "Адапт.", val: prediction.signals.adaptScore, color: "#fb923c" },
                    {
                      label: prediction.modSignal ? `Шаг%${prediction.modSignal.M}` : "Цикл шаг",
                      val: prediction.signals.modScore,
                      color: "#f472b6",
                    },
                    {
                      label: prediction.timeSignal ? `Время%${prediction.timeSignal.periodMs}мс` : "Цикл время",
                      val: prediction.signals.timeScore,
                      color: "#fb923c",
                    },
                  ].map(s => (
                    <div key={s.label} className="text-center">
                      <div className="h-1 rounded-full bg-white/5 mb-1 overflow-hidden">
                        <div className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${Math.min(s.val / 0.5 * 100, 100)}%`, background: s.color, opacity: s.val > 0 ? 1 : 0.25 }} />
                      </div>
                      <p className="font-mono text-white/25" style={{ fontSize: 9 }}>{s.label}</p>
                    </div>
                  ))}
                </div>

                {/* Детали mod/time сигналов — если найдены */}
                {(prediction.modSignal || prediction.timeSignal) && (
                  <div className="border-t px-5 py-2 flex flex-wrap gap-3" style={{ borderColor: "rgba(255,255,255,0.04)" }}>
                    {prediction.modSignal && (
                      <div className="flex items-center gap-1.5 font-mono text-xs">
                        <span style={{ color: "#f472b6" }}>◆</span>
                        <span className="text-white/30">
                          каждые {prediction.modSignal.M} раунда, позиция {prediction.modSignal.remainder} →{" "}
                          <span style={{ color: prediction.modSignal.reactor === "alpha" ? "#22d3ee" : "#c084fc" }}>
                            {prediction.modSignal.reactor === "alpha" ? "α" : "ω"}
                          </span>
                          {" "}(n={prediction.modSignal.sampleCount})
                        </span>
                      </div>
                    )}
                    {prediction.timeSignal && (
                      <div className="flex items-center gap-1.5 font-mono text-xs">
                        <span style={{ color: "#fb923c" }}>◆</span>
                        <span className="text-white/30">
                          период {prediction.timeSignal.periodMs}мс, окно {prediction.timeSignal.bucketIdx + 1}/4 →{" "}
                          <span style={{ color: prediction.timeSignal.reactor === "alpha" ? "#22d3ee" : "#c084fc" }}>
                            {prediction.timeSignal.reactor === "alpha" ? "α" : "ω"}
                          </span>
                          {" "}(n={prediction.timeSignal.sampleCount})
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* ── ПАМЯТЬ ИИ ── */}
            <div className="glass-card rounded-xl overflow-hidden border" style={{ borderColor: "rgba(139,92,246,0.2)" }}>
              <div className="flex items-center justify-between px-4 py-3">
                <div className="flex items-center gap-2">
                  <Icon name="Brain" size={13} className="text-violet-400" />
                  <span className="font-display text-xs tracking-widest text-violet-400/80 uppercase">Память ИИ</span>
                  {memoryMeta && (
                    <span className="font-mono text-xs text-white/25">
                      · {memoryMeta.rounds} раундов · {memoryMeta.hypothesesCount} гипотез
                      {memoryStatus === "loaded" && <span className="text-emerald-400/70"> · загружено</span>}
                      {memoryStatus === "saved" && <span className="text-violet-400/70"> · сохранено {new Date(memoryMeta.savedAt).toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })}</span>}
                    </span>
                  )}
                  {memoryStatus === "none" && !memoryMeta && (
                    <span className="font-mono text-xs text-white/20">· нет сохранений</span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleSaveMemory}
                    className="flex items-center gap-1.5 font-mono text-xs px-3 py-1.5 rounded-lg transition-all"
                    style={{ background: "rgba(139,92,246,0.15)", color: "#c4b5fd", border: "1px solid rgba(139,92,246,0.3)" }}
                    title="Сохранить обученные гипотезы в браузер"
                  >
                    <Icon name="Save" size={11} />
                    Сохранить
                  </button>
                  {memoryStatus === "saved" && (
                    <button
                      onClick={handleLoadMemory}
                      className="flex items-center gap-1.5 font-mono text-xs px-3 py-1.5 rounded-lg transition-all"
                      style={{ background: "rgba(52,211,153,0.1)", color: "#6ee7b7", border: "1px solid rgba(52,211,153,0.25)" }}
                      title="Загрузить ранее сохранённые гипотезы"
                    >
                      <Icon name="Download" size={11} />
                      Загрузить
                    </button>
                  )}
                  {memoryStatus !== "none" && (
                    <button
                      onClick={handleClearMemory}
                      className="flex items-center gap-1.5 font-mono text-xs px-3 py-1.5 rounded-lg transition-all"
                      style={{ background: "rgba(248,113,113,0.08)", color: "#fca5a5", border: "1px solid rgba(248,113,113,0.2)" }}
                      title="Сбросить память и начать обучение заново"
                    >
                      <Icon name="Trash2" size={11} />
                      Сброс
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* ── ИИ-ПРЕДСКАЗАТЕЛЬ ── */}
            {(aiPrediction || aiThoughts.length > 0) && (
              <div className="glass-card rounded-xl overflow-hidden border"
                style={{ borderColor: "rgba(139,92,246,0.25)", background: "rgba(139,92,246,0.04)" }}>

                {/* Заголовок */}
                <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: "rgba(139,92,246,0.15)" }}>
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-violet-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-violet-500"></span>
                    </span>
                    <span className="font-display text-xs tracking-widest text-violet-400 uppercase">ИИ — обучение в реальном времени</span>
                  </div>
                  {aiPrediction && (
                    <div className="font-mono text-xs text-white/30">
                      прогресс: <span className="text-violet-400">{Math.round(aiPrediction.learningProgress * 100)}%</span>
                    </div>
                  )}
                </div>

                {/* Прогноз ИИ */}
                {aiPrediction?.reactor && (
                  <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: "rgba(255,255,255,0.04)" }}>
                    <div>
                      <p className="font-mono text-xs text-white/25 uppercase tracking-widest mb-1.5">ИИ предсказывает</p>
                      <ReactorBadge reactor={aiPrediction.reactor} size="lg" />
                      {aiPrediction.dominantFeature && (
                        <p className="font-mono text-xs text-violet-300/50 mt-1.5">▶ {aiPrediction.dominantFeature}</p>
                      )}
                    </div>
                    <div className="text-center">
                      <p className="font-display text-4xl" style={{ color: "#a78bfa", textShadow: "0 0 20px #a78bfa88" }}>
                        {Math.round(aiPrediction.confidence * 100)}%
                      </p>
                      <p className="font-mono text-xs text-white/25 mt-0.5">уверенность</p>
                    </div>
                  </div>
                )}

                {/* Активные гипотезы */}
                {aiPrediction?.activeHypotheses && aiPrediction.activeHypotheses.length > 0 && (
                  <div className="px-4 py-2 border-b" style={{ borderColor: "rgba(255,255,255,0.04)" }}>
                    <p className="font-mono text-white/25 mb-1.5" style={{ fontSize: 9, letterSpacing: "0.1em" }}>АКТИВНЫЕ ГИПОТЕЗЫ</p>
                    <div className="flex flex-wrap gap-1.5">
                      {aiPrediction.activeHypotheses.map((h, i) => (
                        <span key={i} className="font-mono text-xs px-2 py-0.5 rounded-md"
                          style={{ background: "rgba(139,92,246,0.12)", color: "#c4b5fd", border: "1px solid rgba(139,92,246,0.2)" }}>
                          {h}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Мысли ИИ */}
                {aiThoughts.length > 0 && (
                  <div className="px-4 py-3">
                    <p className="font-mono text-white/25 mb-2" style={{ fontSize: 9, letterSpacing: "0.1em" }}>ВНУТРЕННИЙ МОНОЛОГ ИИ</p>
                    <div className="space-y-1.5 max-h-48 overflow-y-auto" style={{ scrollbarWidth: "none" }}>
                      {[...aiThoughts].reverse().map(t => {
                        const icons: Record<AIThought["type"], string> = {
                          observe: "👁",
                          hypothesis: "💡",
                          correct: "↩",
                          doubt: "⚠",
                          confirm: "✓",
                          discover: "★",
                        };
                        const colors: Record<AIThought["type"], string> = {
                          observe: "rgba(148,163,184,0.7)",
                          hypothesis: "rgba(167,139,250,0.9)",
                          correct: "rgba(251,191,36,0.85)",
                          doubt: "rgba(248,113,113,0.8)",
                          confirm: "rgba(52,211,153,0.9)",
                          discover: "rgba(251,146,60,0.9)",
                        };
                        return (
                          <div key={t.id} className="flex items-start gap-2 font-mono text-xs leading-relaxed">
                            <span className="flex-shrink-0 w-4 text-center opacity-70" style={{ fontSize: 10 }}>{icons[t.type]}</span>
                            <span style={{ color: colors[t.type] }}>{t.text}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Прогресс обучения */}
                {aiPrediction && (
                  <div className="px-4 pb-3">
                    <div className="h-1 rounded-full bg-white/5 overflow-hidden">
                      <div className="h-full rounded-full transition-all duration-700"
                        style={{
                          width: `${aiPrediction.learningProgress * 100}%`,
                          background: "linear-gradient(90deg, #7c3aed, #a78bfa, #c4b5fd)",
                          boxShadow: "0 0 8px #a78bfa55",
                        }}
                      />
                    </div>
                    <p className="font-mono text-white/20 mt-1" style={{ fontSize: 9 }}>
                      накоплено данных · нужно ~20 раундов для полного обучения
                    </p>
                  </div>
                )}
              </div>
            )}

          </div>
          </div> {/* конец левой колонки */}

          {/* ── ПРАВАЯ КОЛОНКА: История событий ── */}
          <div className="glass-card rounded-xl flex flex-col sticky top-28" style={{ height: "calc(100vh - 180px)", minHeight: 500 }}>
            <div className="px-4 pt-4 pb-3 border-b border-white/5 flex-shrink-0">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Icon name="Clock" size={13} className="text-neon-green" />
                  <span className="font-display text-xs tracking-widest text-neon-green/80 uppercase">История событий</span>
                </div>
                <div className="flex items-center gap-2 font-mono text-xs">
                  <span className="text-cyan-400">α {alphaWins}</span>
                  <span className="text-white/20">·</span>
                  <span className="text-purple-400">ω {omegaWins}</span>
                  <span className="text-white/20">·</span>
                  <span className="text-white/40">{totalRounds} всего</span>
                </div>
              </div>
              {/* Точность прогноза */}
              {(() => {
                const withPred = history.filter(r => r.predictionHit !== null);
                const hits = withPred.filter(r => r.predictionHit).length;
                const acc = withPred.length > 0 ? hits / withPred.length : null;
                return withPred.length > 0 ? (
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono text-xs text-white/30">Точность прогноза</span>
                      <span className="font-mono text-xs font-bold" style={{ color: acc! >= 0.6 ? "#00ffcc" : acc! >= 0.4 ? "#facc15" : "#f43f5e" }}>
                        {hits}/{withPred.length} — {Math.round(acc! * 100)}%
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                      <div className="h-full rounded-full transition-all duration-700"
                        style={{
                          width: `${acc! * 100}%`,
                          background: acc! >= 0.6 ? "linear-gradient(90deg,#00ffcc,#38bdf8)" : acc! >= 0.4 ? "#facc15" : "#f43f5e",
                        }}
                      />
                    </div>
                  </div>
                ) : (
                  <div className="font-mono text-xs text-white/20">Точность появится после 2+ раундов</div>
                );
              })()}
            </div>

            {history.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center py-10 gap-2">
                <Icon name="Clock" size={28} className="text-white/10" />
                <p className="font-mono text-xs text-white/25">Событий пока нет</p>
              </div>
            ) : (
              <div className="overflow-y-auto flex-1 p-3 space-y-1.5">
                {[...history].reverse().map((r, i) => (
                  <div
                    key={r.id}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-mono transition-all"
                    style={{
                      background: i === 0 ? (r.winner === "alpha" ? "rgba(34,211,238,0.08)" : "rgba(192,132,252,0.08)") : "rgba(255,255,255,0.02)",
                      border: `1px solid ${i === 0 ? (r.winner === "alpha" ? "rgba(34,211,238,0.25)" : "rgba(192,132,252,0.25)") : "transparent"}`,
                    }}
                  >
                    <span className="text-white/20 w-5 flex-shrink-0">#{r.id}</span>
                    <span className="font-bold w-14 flex-shrink-0" style={{ color: r.winner === "alpha" ? "#22d3ee" : "#c084fc" }}>
                      {r.winner === "alpha" ? "α Альфа" : "ω Омега"}
                    </span>
                    <span className="text-white/30 flex-1">{fmt(r.timestamp)}</span>
                    {/* Прогноз до раунда */}
                    {r.predictedBefore !== null ? (
                      <span
                        className="flex items-center gap-1 flex-shrink-0 px-1.5 py-0.5 rounded text-xs font-bold"
                        style={{
                          background: r.predictionHit ? "rgba(0,255,204,0.12)" : "rgba(244,63,94,0.12)",
                          border: `1px solid ${r.predictionHit ? "rgba(0,255,204,0.3)" : "rgba(244,63,94,0.3)"}`,
                          color: r.predictionHit ? "#00ffcc" : "#f43f5e",
                        }}
                      >
                        {r.predictionHit ? "✓" : "✗"}
                        <span style={{ color: r.predictedBefore === "alpha" ? "#22d3ee" : "#c084fc" }}>
                          {r.predictedBefore === "alpha" ? "α" : "ω"}
                        </span>
                      </span>
                    ) : (
                      <span className="text-white/15 text-xs flex-shrink-0">—</span>
                    )}
                    {i === 0 && (
                      <span className="text-neon-green animate-blink text-xs">▌</span>
                    )}
                  </div>
                ))}
              </div>
            )}

            {totalRounds > 0 && (
              <div className="px-4 py-3 border-t border-white/5 flex-shrink-0">
                <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{ width: `${(alphaWins / totalRounds) * 100}%`, background: "linear-gradient(90deg, #22d3ee, #c084fc)" }}
                  />
                </div>
                <div className="flex justify-between mt-1">
                  <span className="font-mono text-xs text-cyan-400/60">α {Math.round(alphaWins / totalRounds * 100)}%</span>
                  <span className="font-mono text-xs text-purple-400/60">ω {Math.round(omegaWins / totalRounds * 100)}%</span>
                </div>
              </div>
            )}
          </div>
          </div>
        )}



        {/* ── МОДЕЛЬ ML ── */}
        {tab === "model" && (
          <div className="space-y-5 animate-fade-in-up">
            <div className="grid sm:grid-cols-3 gap-4">
              {[
                { label: "Алгоритм", value: "Markov", sub: "+ Flicker Analysis", color: "text-neon-green", icon: "Cpu" },
                { label: "Окно паттернов", value: "2–5", sub: "последних раундов", color: "text-cyan-400", icon: "GitBranch" },
                { label: "Раундов обработано", value: totalRounds, sub: "из захвата экрана", color: "text-purple-400", icon: "Database" },
              ].map(s => (
                <div key={s.label} className="glass-card rounded-xl p-5 text-center">
                  <Icon name={s.icon} size={20} className={`${s.color} mx-auto mb-3`} />
                  <p className={`font-display text-2xl ${s.color} mb-1`}>{s.value}</p>
                  <p className="text-white/40 text-xs font-mono">{s.label}</p>
                  <p className="text-white/25 text-xs mt-0.5">{s.sub}</p>
                </div>
              ))}
            </div>

            <div className="glass-card rounded-xl p-5">
              <div className="flex items-center gap-2 mb-4">
                <Icon name="Settings" size={14} className="text-neon-green" />
                <span className="font-display text-xs tracking-widest text-neon-green/80 uppercase">Как работает предсказание</span>
              </div>
              <div className="space-y-4">
                {[
                  {
                    step: "01", title: "Захват и анализ пикселей",
                    desc: "Каждые ~100мс программа анализирует левую (Альфа) и правую (Омега) половины захваченного окна. Считает долю жёлтых пикселей (R>180, G>160, B<80).",
                    color: "#00ffcc",
                  },
                  {
                    step: "02", title: "Детекция мерцания",
                    desc: "Когда жёлтый сигнал превышает порог 4% — фиксируется мерцание. Записывается: какая сторона мерцает, как часто меняются стороны (темп), смещение (α vs ω).",
                    color: "#facc15",
                  },
                  {
                    step: "03", title: "Детекция результата",
                    desc: "При превышении порога 8% жёлтого — фиксируется SUCCESS. Победитель = та сторона, у которой жёлтого больше. Раунд записывается в историю.",
                    color: "#38bdf8",
                  },
                  {
                    step: "04", title: "Поиск паттернов (Марков)",
                    desc: "Ищутся цепочки длиной 2–5 раундов в истории. Для каждой цепочки считается вероятность следующего исхода. Выбирается самый уверенный совпавший паттерн.",
                    color: "#a855f7",
                  },
                  {
                    step: "05", title: "Взвешивание с мерцанием",
                    desc: "Смещение мерцания добавляет/убирает вес к паттерну. Логика: та сторона, которая мерцала МЕНЬШЕ — чаще побеждает. Вес мерцания: до 40% от уверенности.",
                    color: "#fb923c",
                  },
                ].map(s => (
                  <div key={s.step} className="flex gap-4">
                    <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 font-display text-xs" style={{ background: `${s.color}15`, border: `1px solid ${s.color}30`, color: s.color }}>
                      {s.step}
                    </div>
                    <div>
                      <p className="font-display text-sm tracking-wide text-white/80 mb-1">{s.title}</p>
                      <p className="font-mono text-xs text-white/40 leading-relaxed">{s.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="glass-card rounded-xl p-5">
              <div className="flex items-center gap-2 mb-4">
                <Icon name="Lightbulb" size={14} className="text-yellow-400" />
                <span className="font-display text-xs tracking-widest text-yellow-400/80 uppercase">Важные замечания</span>
              </div>
              <ul className="space-y-2">
                {[
                  "Анализ пикселей работает только если область захвата содержит именно две колонки реакторов без лишних элементов",
                  "Чем больше раундов — тем точнее паттерны. Рекомендуется минимум 10 раундов перед использованием предсказаний",
                  "Темп мерцания влияет на результат: быстрое хаотичное мерцание может снижать уверенность предсказания",
                  "Алгоритм не гарантирует правильность — он ищет статистические закономерности в уже случившихся раундах",
                ].map((t, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-white/45 font-mono">
                    <span className="text-yellow-400 mt-0.5 flex-shrink-0">◆</span>
                    <span>{t}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}