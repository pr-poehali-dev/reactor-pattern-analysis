import { useState, useEffect, useRef, useCallback } from "react";
import Icon from "@/components/ui/icon";
import { analyzeFrame, computeFlickerStats } from "@/lib/screenAnalyzer";
import { predict, getTopPatterns } from "@/lib/mlPredictor";
import type { RoundResult, FlickerSample, FrameAnalysis } from "@/lib/screenAnalyzer";
import type { Prediction, Pattern } from "@/lib/mlPredictor";

type Tab = "capture" | "history" | "prediction" | "stats" | "model";

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
  const [capturing, setCapturing] = useState(false);
  const [history, setHistory] = useState<RoundResult[]>([]);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [lastFrame, setLastFrame] = useState<FrameAnalysis | null>(null);
  const [flickerSamples, setFlickerSamples] = useState<FlickerSample[]>([]);
  const [roundPhase, setRoundPhase] = useState<"idle" | "flicker" | "result">("idle");
  const [countdown, setCountdown] = useState<number>(30);
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [liveTime, setLiveTime] = useState(new Date());
  const [captureError, setCaptureError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
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

  const startCapture = useCallback(async () => {
    setCaptureError(null);
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 10 },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      stream.getVideoTracks()[0].addEventListener("ended", stopCapture);
      setCapturing(true);
    } catch {
      setCaptureError("Не удалось получить доступ к экрану. Разрешите захват окна в браузере.");
    }
  }, []);

  const stopCapture = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (analyzeLoopRef.current) cancelAnimationFrame(analyzeLoopRef.current);
    setCapturing(false);
    phaseRef.current = "idle";
    setRoundPhase("idle");
    flickerBufRef.current = [];
  }, []);

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

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 360;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const frame = analyzeFrame(ctx, canvas.width, canvas.height);
      setLastFrame(frame);

      const now = frame.timestamp;
      const prevPhase = phaseRef.current;

      // Детектируем мерцание
      if (frame.phase === "flicker") {
        phaseRef.current = "flicker";
        setRoundPhase("flicker");
        const dominant = frame.alphaYellow >= frame.omegaYellow ? "alpha" : "omega";
        flickerBufRef.current.push({ timestamp: now, dominant });
        // Оставляем только последние 6 секунд
        flickerBufRef.current = flickerBufRef.current.filter(s => now - s.timestamp < 6000);
        setFlickerSamples([...flickerBufRef.current]);
      }

      // Детектируем результат (стабильный жёлтый > 0.5 сек после последнего результата)
      if (frame.phase === "result" && frame.winner && now - lastResultTsRef.current > 15000) {
        lastResultTsRef.current = now;
        phaseRef.current = "result";
        setRoundPhase("result");

        const flickerStats = computeFlickerStats(flickerBufRef.current);
        const newResult: RoundResult = {
          id: roundIdRef.current++,
          winner: frame.winner,
          timestamp: now,
          flickerPattern: [...flickerBufRef.current],
          flickerRate: flickerStats.rate,
          flickerBias: flickerStats.bias,
        };

        setHistory(prev => {
          const next = [...prev, newResult];
          const pred = predict(next, flickerStats.bias, flickerStats.rate);
          setPrediction(pred);
          setPatterns(getTopPatterns(next));
          return next;
        });

        flickerBufRef.current = [];
        setFlickerSamples([]);

        // Через 3 сек возвращаемся в idle
        setTimeout(() => {
          phaseRef.current = "idle";
          setRoundPhase("idle");
        }, 3000);
      }

      // Если фаза была flicker, а теперь idle — обновить предсказание с учётом мерцания
      if (prevPhase === "flicker" && frame.phase === "idle" && flickerBufRef.current.length > 3) {
        const flickerStats = computeFlickerStats(flickerBufRef.current);
        setHistory(prev => {
          const pred = predict(prev, flickerStats.bias, flickerStats.rate);
          setPrediction(pred);
          return prev;
        });
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
    { id: "history", icon: "Clock", label: "История" },
    { id: "prediction", icon: "Zap", label: "Предсказание" },
    { id: "stats", icon: "BarChart2", label: "Статистика" },
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

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-5">

        {/* ── ЗАХВАТ ЭКРАНА ── */}
        {tab === "capture" && (
          <div className="space-y-5 animate-fade-in-up">

            {/* Управление захватом */}
            <div className="glass-card rounded-xl p-5">
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex-1 min-w-0">
                  <h2 className="font-display text-sm tracking-widest text-white/60 uppercase mb-1">Захват области реакторов</h2>
                  <p className="font-mono text-xs text-white/30">
                    Нажми «Начать захват» → выбери окно игры → выдели область с двумя реакторами
                  </p>
                </div>
                <div className="flex gap-3">
                  {!capturing ? (
                    <button
                      onClick={startCapture}
                      className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-display tracking-widest text-sm transition-all duration-300"
                      style={{ background: "linear-gradient(135deg, #00ffcc, #38bdf8)", color: "#080d14", boxShadow: "0 0 20px rgba(0,255,204,0.3)" }}
                    >
                      <Icon name="Monitor" size={15} />
                      Начать захват
                    </button>
                  ) : (
                    <button
                      onClick={stopCapture}
                      className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-display tracking-widest text-sm border border-red-500/40 text-red-400 hover:bg-red-500/10 transition-all"
                    >
                      <Icon name="Square" size={15} />
                      Остановить
                    </button>
                  )}
                </div>
              </div>
              {captureError && (
                <div className="mt-3 px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 font-mono text-xs">
                  {captureError}
                </div>
              )}
            </div>

            {/* Превью захвата */}
            <div className="glass-card rounded-xl overflow-hidden">
              <div className="px-5 pt-4 pb-2 flex items-center justify-between">
                <span className="font-display text-xs tracking-widest text-white/40 uppercase">Превью захваченного экрана</span>
                {capturing && (
                  <span className="flex items-center gap-1.5 text-xs font-mono text-red-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
                    REC
                  </span>
                )}
              </div>
              <div className="relative bg-black/40 mx-4 mb-4 rounded-lg overflow-hidden" style={{ minHeight: 240 }}>
                {capturing ? (
                  <>
                    <video ref={videoRef} className="w-full h-full object-contain" muted playsInline />
                    <canvas ref={canvasRef} className="hidden" />
                    {/* Оверлей разделителя */}
                    <div className="absolute inset-0 pointer-events-none">
                      <div className="absolute top-0 bottom-0 left-1/2 w-px bg-white/20" />
                      <div className="absolute top-2 left-4 text-xs font-display text-cyan-400/70 tracking-widest">α АЛЬФА</div>
                      <div className="absolute top-2 right-4 text-xs font-display text-purple-400/70 tracking-widest">ω ОМЕГА</div>
                    </div>
                  </>
                ) : (
                  <div className="flex flex-col items-center justify-center h-60 gap-3">
                    <Icon name="Monitor" size={40} className="text-white/10" />
                    <p className="font-mono text-xs text-white/20">Захват не активен</p>
                  </div>
                )}
              </div>
            </div>

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
                  {roundPhase === "result" && lastFrame.winner === "alpha" && (
                    <div className="mt-3 text-center py-2 rounded-lg bg-neon-green/10 border border-neon-green/30">
                      <span className="font-display text-sm text-neon-green tracking-widest">✓ SUCCESS</span>
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
                  {roundPhase === "result" && lastFrame.winner === "omega" && (
                    <div className="mt-3 text-center py-2 rounded-lg bg-purple-500/10 border border-purple-500/30">
                      <span className="font-display text-sm text-purple-300 tracking-widest">✓ SUCCESS</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Мерцание в реальном времени */}
            {roundPhase === "flicker" && flickerSamples.length > 0 && (
              <div className="glass-card rounded-xl p-5">
                <div className="flex items-center gap-2 mb-4">
                  <Icon name="Zap" size={14} className="text-yellow-400" />
                  <span className="font-display text-xs tracking-widest text-yellow-400 uppercase">Анализ мерцания (live)</span>
                </div>
                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div className="text-center">
                    <p className="font-mono text-xs text-white/30 mb-1">Темп</p>
                    <p className="font-display text-xl text-yellow-400">{flickerStats.rate.toFixed(1)}<span className="text-xs text-white/30 ml-1">/сек</span></p>
                  </div>
                  <div className="text-center">
                    <p className="font-mono text-xs text-white/30 mb-1">Альфа %</p>
                    <p className="font-display text-xl text-cyan-400">{Math.round(flickerStats.alphaPct * 100)}%</p>
                  </div>
                  <div className="text-center">
                    <p className="font-mono text-xs text-white/30 mb-1">Омега %</p>
                    <p className="font-display text-xl text-purple-400">{Math.round(flickerStats.omegaPct * 100)}%</p>
                  </div>
                </div>
                {/* Визуализация последних сэмплов */}
                <div className="flex gap-1 flex-wrap">
                  {flickerSamples.slice(-30).map((s, i) => (
                    <div
                      key={i}
                      className="w-4 h-6 rounded-sm transition-all"
                      style={{
                        background: s.dominant === "alpha" ? "rgba(34,211,238,0.7)" : "rgba(192,132,252,0.7)",
                        boxShadow: s.dominant === "alpha" ? "0 0 4px #22d3ee" : "0 0 4px #c084fc",
                      }}
                    />
                  ))}
                </div>
                <div className="flex justify-between mt-1">
                  <span className="font-mono text-xs text-white/20">← раньше</span>
                  <span className="font-mono text-xs text-yellow-400">сейчас →</span>
                </div>
              </div>
            )}

            {/* Быстрое предсказание на главном экране */}
            {prediction && prediction.reactor && (
              <div
                className="rounded-xl p-5 border"
                style={{
                  background: prediction.reactor === "alpha" ? "rgba(34,211,238,0.06)" : "rgba(192,132,252,0.06)",
                  borderColor: prediction.reactor === "alpha" ? "rgba(34,211,238,0.3)" : "rgba(192,132,252,0.3)",
                  boxShadow: prediction.reactor === "alpha" ? "0 0 30px rgba(34,211,238,0.1)" : "0 0 30px rgba(192,132,252,0.1)",
                }}
              >
                <div className="flex items-center justify-between flex-wrap gap-3">
                  <div>
                    <p className="font-mono text-xs text-white/30 uppercase tracking-widest mb-2">Следующий победитель</p>
                    <ReactorBadge reactor={prediction.reactor} size="lg" />
                    <p className="font-mono text-xs text-white/40 mt-2">{prediction.reason}</p>
                  </div>
                  <div className="text-center">
                    <p className="font-display text-4xl" style={{ color: prediction.reactor === "alpha" ? "#22d3ee" : "#c084fc", textShadow: `0 0 20px currentColor` }}>
                      {Math.round(prediction.confidence * 100)}%
                    </p>
                    <p className="font-mono text-xs text-white/30 mt-1">уверенность</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── ИСТОРИЯ ── */}
        {tab === "history" && (
          <div className="space-y-5 animate-fade-in-up">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: "Раундов", value: totalRounds, color: "text-neon-green", icon: "List" },
                { label: "Альфа побед", value: alphaWins, color: "text-cyan-400", icon: "Zap" },
                { label: "Омега побед", value: omegaWins, color: "text-purple-400", icon: "Zap" },
                { label: "% Альфа", value: totalRounds ? `${Math.round(alphaWins / totalRounds * 100)}%` : "—", color: "text-yellow-400", icon: "BarChart2" },
              ].map(s => (
                <div key={s.label} className="glass-card rounded-xl p-4 text-center">
                  <Icon name={s.icon} size={16} className={`${s.color} mx-auto mb-2`} />
                  <p className={`font-display text-2xl ${s.color}`}>{s.value}</p>
                  <p className="text-white/40 text-xs mt-1">{s.label}</p>
                </div>
              ))}
            </div>

            {history.length === 0 ? (
              <div className="glass-card rounded-xl p-10 text-center">
                <Icon name="Clock" size={36} className="text-white/10 mx-auto mb-3" />
                <p className="font-mono text-sm text-white/30">Раундов пока нет — запусти захват экрана</p>
              </div>
            ) : (
              <div className="glass-card rounded-xl p-5">
                <div className="flex items-center gap-2 mb-4">
                  <Icon name="Clock" size={14} className="text-neon-green" />
                  <span className="font-display text-xs tracking-widest text-neon-green/80 uppercase">Таблица раундов</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs font-mono">
                    <thead>
                      <tr className="border-b border-white/10">
                        {["#", "ПОБЕДИТЕЛЬ", "ВРЕМЯ", "ТЕМП МЕЦ.", "СМЕЩЕНИЕ", "МЕЦ. ПОДСКАЗКА"].map(h => (
                          <th key={h} className="text-left py-2 px-3 text-white/30 font-normal tracking-widest">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {[...history].reverse().map((r, i) => {
                        const flickerHint = r.flickerBias > 0.1 ? "omega" : r.flickerBias < -0.1 ? "alpha" : null;
                        return (
                          <tr key={r.id} className={`border-b border-white/5 ${i === 0 ? "bg-neon-green/5" : "hover:bg-white/3"}`}>
                            <td className="py-2.5 px-3 text-white/25">#{r.id}</td>
                            <td className="py-2.5 px-3"><ReactorBadge reactor={r.winner} size="sm" /></td>
                            <td className="py-2.5 px-3 text-white/50">{fmt(r.timestamp)}</td>
                            <td className="py-2.5 px-3 text-yellow-400">{r.flickerRate.toFixed(1)}/с</td>
                            <td className="py-2.5 px-3">
                              <span className={r.flickerBias > 0 ? "text-cyan-400" : r.flickerBias < 0 ? "text-purple-400" : "text-white/30"}>
                                {r.flickerBias > 0 ? `α +${(r.flickerBias * 100).toFixed(0)}%` : r.flickerBias < 0 ? `ω +${(Math.abs(r.flickerBias) * 100).toFixed(0)}%` : "нейтр."}
                              </span>
                            </td>
                            <td className="py-2.5 px-3">
                              {flickerHint ? (
                                <ReactorBadge reactor={flickerHint as "alpha" | "omega"} size="sm" />
                              ) : <span className="text-white/20">—</span>}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── ПРЕДСКАЗАНИЕ ── */}
        {tab === "prediction" && (
          <div className="space-y-5 animate-fade-in-up">
            {!prediction || !prediction.reactor ? (
              <div className="glass-card rounded-xl p-10 text-center">
                <Icon name="Zap" size={36} className="text-white/10 mx-auto mb-3" />
                <p className="font-mono text-sm text-white/30">Нужно минимум 2 раунда для предсказания</p>
              </div>
            ) : (
              <>
                <div className="grid sm:grid-cols-2 gap-5">
                  {/* Предсказание */}
                  <div
                    className="rounded-xl p-6 border scan-line"
                    style={{
                      background: prediction.reactor === "alpha" ? "rgba(34,211,238,0.07)" : "rgba(192,132,252,0.07)",
                      borderColor: prediction.reactor === "alpha" ? "rgba(34,211,238,0.35)" : "rgba(192,132,252,0.35)",
                    }}
                  >
                    <p className="font-mono text-xs text-white/30 uppercase tracking-widest mb-3">Следующий победитель</p>
                    <div className="flex items-center gap-4">
                      <div
                        className="w-24 h-24 rounded-2xl flex items-center justify-center text-4xl font-display border"
                        style={{
                          background: prediction.reactor === "alpha" ? "rgba(34,211,238,0.1)" : "rgba(192,132,252,0.1)",
                          borderColor: prediction.reactor === "alpha" ? "rgba(34,211,238,0.4)" : "rgba(192,132,252,0.4)",
                          color: prediction.reactor === "alpha" ? "#22d3ee" : "#c084fc",
                          boxShadow: prediction.reactor === "alpha" ? "0 0 30px rgba(34,211,238,0.2)" : "0 0 30px rgba(192,132,252,0.2)",
                        }}
                      >
                        {prediction.reactor === "alpha" ? "α" : "ω"}
                      </div>
                      <div>
                        <p className="font-display text-2xl tracking-widest" style={{ color: prediction.reactor === "alpha" ? "#22d3ee" : "#c084fc" }}>
                          {prediction.reactor === "alpha" ? "АЛЬФА" : "ОМЕГА"}
                        </p>
                        <p className="font-mono text-xs text-white/40 mt-1 max-w-xs">{prediction.reason}</p>
                      </div>
                    </div>
                  </div>

                  {/* Уверенность */}
                  <div className="glass-card rounded-xl p-6">
                    <p className="font-mono text-xs text-white/30 uppercase tracking-widest mb-3">Уверенность модели</p>
                    <p className="font-display text-5xl text-neon-green mb-4" style={{ textShadow: "0 0 30px rgba(0,255,204,0.4)" }}>
                      {Math.round(prediction.confidence * 100)}%
                    </p>
                    <ConfBar value={prediction.confidence} color="#00ffcc" />
                    <div className="grid grid-cols-2 gap-3 mt-4">
                      <div className="text-center p-2 rounded-lg bg-white/3">
                        <p className="font-mono text-xs text-white/30 mb-1">Паттерн</p>
                        <p className="font-mono text-xs text-neon-green">{prediction.patternMatch ? `${Math.round(prediction.patternMatch.confidence * 100)}%` : "нет"}</p>
                      </div>
                      <div className="text-center p-2 rounded-lg bg-white/3">
                        <p className="font-mono text-xs text-white/30 mb-1">Мерцание</p>
                        <p className="font-mono text-xs text-yellow-400">{prediction.flickerHint ? `${Math.round(prediction.flickerWeight * 100)}%` : "нет"}</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Паттерн */}
                {prediction.patternMatch && (
                  <div className="glass-card rounded-xl p-5">
                    <div className="flex items-center gap-2 mb-4">
                      <Icon name="GitBranch" size={14} className="text-neon-green" />
                      <span className="font-display text-xs tracking-widest text-neon-green/80 uppercase">Совпавший паттерн</span>
                    </div>
                    <div className="flex items-center gap-2 flex-wrap">
                      {prediction.patternMatch.sequence.map((r, i) => (
                        <div key={i} className="flex items-center gap-2">
                          <div
                            className="px-3 py-2 rounded-lg border text-center min-w-16"
                            style={{
                              background: r === "alpha" ? "rgba(34,211,238,0.1)" : "rgba(192,132,252,0.1)",
                              borderColor: r === "alpha" ? "rgba(34,211,238,0.3)" : "rgba(192,132,252,0.3)",
                            }}
                          >
                            <span className="font-display text-lg" style={{ color: r === "alpha" ? "#22d3ee" : "#c084fc" }}>
                              {r === "alpha" ? "α" : "ω"}
                            </span>
                          </div>
                          {i < prediction.patternMatch!.sequence.length - 1 && (
                            <Icon name="ArrowRight" size={12} className="text-white/20" />
                          )}
                        </div>
                      ))}
                      <Icon name="ArrowRight" size={14} className="text-neon-green/60" />
                      <div
                        className="px-3 py-2 rounded-lg border text-center min-w-16 animate-pulse"
                        style={{
                          background: prediction.reactor === "alpha" ? "rgba(34,211,238,0.15)" : "rgba(192,132,252,0.15)",
                          borderColor: prediction.reactor === "alpha" ? "rgba(34,211,238,0.5)" : "rgba(192,132,252,0.5)",
                        }}
                      >
                        <span className="font-display text-lg" style={{ color: prediction.reactor === "alpha" ? "#22d3ee" : "#c084fc" }}>
                          {prediction.reactor === "alpha" ? "α" : "ω"}
                        </span>
                      </div>
                      <span className="font-mono text-xs text-white/30 ml-2">встречалось {prediction.patternMatch.count}×</span>
                    </div>
                  </div>
                )}

                {/* Последние раунды */}
                <div className="glass-card rounded-xl p-5">
                  <div className="flex items-center gap-2 mb-4">
                    <Icon name="History" size={14} className="text-neon-green" />
                    <span className="font-display text-xs tracking-widest text-neon-green/80 uppercase">Последние раунды (контекст)</span>
                  </div>
                  <div className="flex items-center gap-2 flex-wrap">
                    {history.slice(-10).map((r, i, arr) => (
                      <div key={r.id} className="flex items-center gap-2">
                        <div
                          className="flex flex-col items-center px-2 py-1.5 rounded-lg border text-xs"
                          style={{
                            background: r.winner === "alpha" ? "rgba(34,211,238,0.08)" : "rgba(192,132,252,0.08)",
                            borderColor: r.winner === "alpha" ? "rgba(34,211,238,0.25)" : "rgba(192,132,252,0.25)",
                          }}
                        >
                          <span style={{ color: r.winner === "alpha" ? "#22d3ee" : "#c084fc" }} className="font-display text-base">
                            {r.winner === "alpha" ? "α" : "ω"}
                          </span>
                          <span className="text-white/20 font-mono" style={{ fontSize: 9 }}>#{r.id}</span>
                        </div>
                        {i < arr.length - 1 && <Icon name="ChevronRight" size={10} className="text-white/15" />}
                      </div>
                    ))}
                    <Icon name="ChevronRight" size={12} className="text-neon-green/40" />
                    <div
                      className="px-2 py-1.5 rounded-lg border text-xs animate-pulse"
                      style={{
                        borderStyle: "dashed",
                        borderColor: prediction.reactor === "alpha" ? "rgba(34,211,238,0.5)" : "rgba(192,132,252,0.5)",
                        background: prediction.reactor === "alpha" ? "rgba(34,211,238,0.08)" : "rgba(192,132,252,0.08)",
                      }}
                    >
                      <span style={{ color: prediction.reactor === "alpha" ? "#22d3ee" : "#c084fc" }} className="font-display text-base">
                        {prediction.reactor === "alpha" ? "α" : "ω"}
                      </span>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* ── СТАТИСТИКА ── */}
        {tab === "stats" && (
          <div className="space-y-5 animate-fade-in-up">
            {/* Частота */}
            <div className="grid sm:grid-cols-2 gap-5">
              <div className="glass-card rounded-xl p-5">
                <div className="flex items-center gap-2 mb-4">
                  <Icon name="BarChart2" size={14} className="text-neon-green" />
                  <span className="font-display text-xs tracking-widest text-neon-green/80 uppercase">Распределение побед</span>
                </div>
                <div className="space-y-4">
                  {[
                    { label: "α Альфа", wins: alphaWins, color: "#22d3ee" },
                    { label: "ω Омега", wins: omegaWins, color: "#c084fc" },
                  ].map(r => (
                    <div key={r.label}>
                      <div className="flex justify-between mb-1.5">
                        <span className="font-mono text-sm" style={{ color: r.color }}>{r.label}</span>
                        <span className="font-mono text-sm text-white/60">{r.wins} раундов · {totalRounds ? Math.round(r.wins / totalRounds * 100) : 0}%</span>
                      </div>
                      <ConfBar value={totalRounds ? r.wins / totalRounds : 0} color={r.color} />
                    </div>
                  ))}
                </div>
                {totalRounds > 0 && (
                  <div className="mt-4 h-6 rounded-full overflow-hidden flex">
                    <div style={{ width: `${alphaWins / totalRounds * 100}%`, background: "#22d3ee55" }} />
                    <div style={{ flex: 1, background: "#c084fc55" }} />
                  </div>
                )}
              </div>

              {/* Мерцание по раундам */}
              <div className="glass-card rounded-xl p-5">
                <div className="flex items-center gap-2 mb-4">
                  <Icon name="Zap" size={14} className="text-yellow-400" />
                  <span className="font-display text-xs tracking-widest text-yellow-400/80 uppercase">Анализ мерцания по раундам</span>
                </div>
                {history.length === 0 ? (
                  <p className="font-mono text-xs text-white/25 text-center py-6">Нет данных</p>
                ) : (
                  <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                    {[...history].reverse().slice(0, 15).map(r => (
                      <div key={r.id} className="flex items-center gap-2 text-xs font-mono">
                        <span className="text-white/25 w-8">#{r.id}</span>
                        <ReactorBadge reactor={r.winner} size="sm" />
                        <div className="flex-1 h-4 rounded bg-white/5 overflow-hidden flex">
                          <div style={{ width: `${r.flickerBias > 0 ? r.flickerBias * 100 : 0}%`, background: "#22d3ee55" }} />
                          <div style={{ width: `${r.flickerBias < 0 ? Math.abs(r.flickerBias) * 100 : 0}%`, background: "#c084fc55" }} />
                        </div>
                        <span className="text-yellow-400 w-12 text-right">{r.flickerRate.toFixed(1)}/с</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Паттерны */}
            <div className="glass-card rounded-xl p-5">
              <div className="flex items-center gap-2 mb-4">
                <Icon name="GitMerge" size={14} className="text-neon-green" />
                <span className="font-display text-xs tracking-widest text-neon-green/80 uppercase">Найденные паттерны последовательностей</span>
              </div>
              {patterns.length === 0 ? (
                <p className="font-mono text-xs text-white/25 text-center py-6">Нужно больше раундов для поиска паттернов</p>
              ) : (
                <div className="grid sm:grid-cols-2 gap-3">
                  {patterns.map((p, i) => (
                    <div key={i} className="glass-card-purple rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <span className="font-mono text-xs text-white/40">{p.label}</span>
                        <span className="font-mono text-xs text-purple-400">{Math.round(p.confidence * 100)}%</span>
                      </div>
                      <div className="flex items-center gap-1.5 mb-2 flex-wrap">
                        {p.sequence.map((r, j) => (
                          <div key={j} className="flex items-center gap-1">
                            <span
                              className="w-7 h-7 flex items-center justify-center rounded font-display text-sm border"
                              style={{
                                background: r === "alpha" ? "rgba(34,211,238,0.12)" : "rgba(192,132,252,0.12)",
                                borderColor: r === "alpha" ? "rgba(34,211,238,0.3)" : "rgba(192,132,252,0.3)",
                                color: r === "alpha" ? "#22d3ee" : "#c084fc",
                              }}
                            >{r === "alpha" ? "α" : "ω"}</span>
                            {j < p.sequence.length - 1 && <Icon name="ArrowRight" size={8} className="text-white/20" />}
                          </div>
                        ))}
                        <Icon name="ArrowRight" size={10} className="text-neon-green/40" />
                        <span
                          className="w-7 h-7 flex items-center justify-center rounded font-display text-sm border font-bold"
                          style={{
                            background: p.next === "alpha" ? "rgba(34,211,238,0.2)" : "rgba(192,132,252,0.2)",
                            borderColor: p.next === "alpha" ? "#22d3ee" : "#c084fc",
                            color: p.next === "alpha" ? "#22d3ee" : "#c084fc",
                          }}
                        >{p.next === "alpha" ? "α" : "ω"}</span>
                      </div>
                      <ConfBar value={p.confidence} color={p.next === "alpha" ? "#22d3ee" : "#c084fc"} />
                      <p className="font-mono text-xs text-white/20 mt-1.5">встречается {p.count}×</p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Тепловая карта */}
            {history.length > 0 && (
              <div className="glass-card rounded-xl p-5">
                <div className="flex items-center gap-2 mb-4">
                  <Icon name="Flame" size={14} className="text-orange-400" />
                  <span className="font-display text-xs tracking-widest text-orange-400/80 uppercase">Хронология раундов</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {history.map((r, i) => (
                    <div
                      key={r.id}
                      title={`#${r.id}: ${r.winner === "alpha" ? "Альфа" : "Омега"} · ${fmt(r.timestamp)}`}
                      className="w-7 h-7 rounded-md flex items-center justify-center cursor-default transition-transform hover:scale-110 font-display text-xs"
                      style={{
                        background: r.winner === "alpha" ? `rgba(34,211,238,${0.2 + (i / history.length) * 0.5})` : `rgba(192,132,252,${0.2 + (i / history.length) * 0.5})`,
                        border: `1px solid ${r.winner === "alpha" ? "rgba(34,211,238,0.3)" : "rgba(192,132,252,0.3)"}`,
                        color: r.winner === "alpha" ? "#22d3ee" : "#c084fc",
                      }}
                    >
                      {r.winner === "alpha" ? "α" : "ω"}
                    </div>
                  ))}
                </div>
              </div>
            )}
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
