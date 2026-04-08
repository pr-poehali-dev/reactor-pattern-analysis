import { useState, useEffect, useRef } from "react";
import Icon from "@/components/ui/icon";

const REACTORS = ["R-01", "R-02", "R-03", "R-04", "R-05", "R-06", "R-07", "R-08"];
const COLUMNS = ["A", "B", "C", "D", "E", "F"];

type Tab = "monitor" | "history" | "prediction" | "stats" | "model";

interface SelectionEvent {
  id: number;
  reactor: string;
  column: string;
  timestamp: Date;
  interval: number | null;
}

function generateEvent(id: number, prev: SelectionEvent | null): SelectionEvent {
  const reactor = REACTORS[Math.floor(Math.random() * REACTORS.length)];
  const column = COLUMNS[Math.floor(Math.random() * COLUMNS.length)];
  const now = new Date();
  const interval = prev ? Math.round((now.getTime() - prev.timestamp.getTime()) / 1000) : null;
  return { id, reactor, column, timestamp: now, interval };
}

const INITIAL_EVENTS: SelectionEvent[] = (() => {
  const evts: SelectionEvent[] = [];
  const now = Date.now();
  for (let i = 0; i < 20; i++) {
    const ts = new Date(now - (20 - i) * (3000 + Math.random() * 5000));
    evts.push({
      id: i + 1,
      reactor: REACTORS[Math.floor(Math.random() * REACTORS.length)],
      column: COLUMNS[Math.floor(Math.random() * COLUMNS.length)],
      timestamp: ts,
      interval: i === 0 ? null : Math.round((ts.getTime() - (evts[i - 1]?.timestamp.getTime() ?? ts.getTime())) / 1000),
    });
  }
  return evts;
})();

const PATTERNS = [
  { seq: ["A", "B", "C"], confidence: 0.82, count: 14, label: "Восходящий" },
  { seq: ["C", "C", "D"], confidence: 0.71, count: 9, label: "Повтор-сдвиг" },
  { seq: ["F", "E", "D", "C"], confidence: 0.65, count: 6, label: "Нисходящий" },
  { seq: ["A", "D", "A"], confidence: 0.58, count: 5, label: "Зеркальный" },
];

function fmt(d: Date) {
  return d.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function StatusDot({ active }: { active: boolean }) {
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full ${active ? "bg-neon-green animate-pulse-glow" : "bg-gray-600"}`}
      style={active ? { boxShadow: "0 0 8px #00ffcc" } : {}}
    />
  );
}

function SectionCard({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`glass-card rounded-xl p-5 ${className}`}>{children}</div>
  );
}

function SectionTitle({ icon, label }: { icon: string; label: string }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <Icon name={icon} size={16} className="text-neon-green" />
      <h3 className="font-display text-sm tracking-widest uppercase text-neon-green opacity-80">{label}</h3>
    </div>
  );
}

export default function Index() {
  const [tab, setTab] = useState<Tab>("monitor");
  const [events, setEvents] = useState<SelectionEvent[]>(INITIAL_EVENTS);
  const [activeReactor, setActiveReactor] = useState<string>("R-03");
  const [activeColumn, setActiveColumn] = useState<string>("B");
  const [tick, setTick] = useState(0);
  const idRef = useRef(INITIAL_EVENTS.length + 1);
  const [liveTime, setLiveTime] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => {
      setLiveTime(new Date());
      setTick((t) => t + 1);
      if (Math.random() < 0.35) {
        setEvents((prev) => {
          const newEvt = generateEvent(idRef.current++, prev[prev.length - 1] ?? null);
          setActiveReactor(newEvt.reactor);
          setActiveColumn(newEvt.column);
          return [...prev.slice(-49), newEvt];
        });
      }
    }, 1800);
    return () => clearInterval(interval);
  }, []);

  const colCounts = COLUMNS.reduce((acc, c) => {
    acc[c] = events.filter((e) => e.column === c).length;
    return acc;
  }, {} as Record<string, number>);
  const maxColCount = Math.max(...Object.values(colCounts), 1);

  const reactorCounts = REACTORS.reduce((acc, r) => {
    acc[r] = events.filter((e) => e.reactor === r).length;
    return acc;
  }, {} as Record<string, number>);
  const totalEvents = events.length;

  const lastEvent = events[events.length - 1];
  const predictedCol = COLUMNS[(COLUMNS.indexOf(lastEvent?.column ?? "A") + 1) % COLUMNS.length];
  const predictedReactor = lastEvent?.reactor ?? "R-01";
  const confidence = 0.72 + Math.sin(tick * 0.3) * 0.08;

  const intervals = events.filter((e) => e.interval !== null).map((e) => e.interval as number);
  const avgInterval = intervals.length > 0 ? Math.round(intervals.reduce((a, b) => a + b, 0) / intervals.length) : 0;

  const TABS: { id: Tab; icon: string; label: string }[] = [
    { id: "monitor", icon: "Activity", label: "Мониторинг" },
    { id: "history", icon: "Clock", label: "История" },
    { id: "prediction", icon: "Zap", label: "Предсказание" },
    { id: "stats", icon: "BarChart2", label: "Статистика" },
    { id: "model", icon: "Cpu", label: "Модель ML" },
  ];

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
              <div>
                <span className="font-display text-white text-lg tracking-widest">REACTOR<span className="text-neon-green">OS</span></span>
                <span className="hidden sm:inline ml-3 font-mono text-xs text-white/30 tracking-widest">v2.4.1</span>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="hidden md:flex items-center gap-2 text-xs font-mono text-white/40">
                <StatusDot active={true} />
                <span className="text-neon-green/70">{fmt(liveTime)}</span>
              </div>
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-neon-green/10 border border-neon-green/20">
                <StatusDot active={true} />
                <span className="text-neon-green text-xs font-mono font-medium">ONLINE</span>
              </div>
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-neon-purple/10 border border-neon-purple/20">
                <Icon name="Database" size={12} className="text-neon-purple" />
                <span className="text-neon-purple text-xs font-mono">{totalEvents} evt</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Tabs */}
      <div className="sticky top-14 z-40 bg-[#080d14]/80 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="flex gap-0 overflow-x-auto">
            {TABS.map((t) => (
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

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6">

        {/* ── МОНИТОРИНГ ── */}
        {tab === "monitor" && (
          <div className="space-y-5 animate-fade-in-up">
            <div className="glass-card rounded-xl px-6 py-4 flex flex-wrap items-center gap-6">
              <div>
                <p className="text-white/30 text-xs font-mono uppercase tracking-widest mb-1">Активный реактор</p>
                <p className="font-display text-3xl text-neon-green" style={{ textShadow: "0 0 20px rgba(0,255,204,0.5)" }}>{activeReactor}</p>
              </div>
              <div className="w-px h-10 bg-white/10" />
              <div>
                <p className="text-white/30 text-xs font-mono uppercase tracking-widest mb-1">Колонка</p>
                <p className="font-display text-3xl text-neon-purple" style={{ textShadow: "0 0 20px rgba(168,85,247,0.5)" }}>{activeColumn}</p>
              </div>
              <div className="w-px h-10 bg-white/10" />
              <div>
                <p className="text-white/30 text-xs font-mono uppercase tracking-widest mb-1">Последнее событие</p>
                <p className="font-mono text-sm text-white/70">{lastEvent ? fmt(lastEvent.timestamp) : "—"}</p>
              </div>
              <div className="ml-auto">
                <div className="w-3 h-3 rounded-full bg-neon-green animate-pulse" style={{ boxShadow: "0 0 15px #00ffcc" }} />
              </div>
            </div>

            <SectionCard>
              <SectionTitle icon="Server" label="Реакторы — уровень активности" />
              <div className="grid grid-cols-4 sm:grid-cols-8 gap-2">
                {REACTORS.map((r) => {
                  const active = r === activeReactor;
                  const pct = Math.round((reactorCounts[r] / totalEvents) * 100);
                  const maxRCount = Math.max(...Object.values(reactorCounts), 1);
                  return (
                    <div
                      key={r}
                      className={`relative rounded-lg border p-3 text-center transition-all duration-500 cursor-default ${
                        active ? "reactor-col-active" : "reactor-col-inactive"
                      }`}
                    >
                      <p className={`font-display text-xs tracking-widest mb-2 ${active ? "text-neon-green" : "text-white/40"}`}>{r}</p>
                      <div className="h-16 flex flex-col justify-end mb-2">
                        <div
                          className="w-full rounded-sm transition-all duration-700"
                          style={{
                            height: `${Math.max(8, (reactorCounts[r] / maxRCount) * 64)}px`,
                            background: active
                              ? "linear-gradient(180deg, #00ffcc, #00ffcc44)"
                              : "linear-gradient(180deg, rgba(56,189,248,0.5), rgba(56,189,248,0.1))",
                            boxShadow: active ? "0 0 12px rgba(0,255,204,0.4)" : "none",
                          }}
                        />
                      </div>
                      <p className={`font-mono text-xs ${active ? "text-neon-green" : "text-white/30"}`}>{pct}%</p>
                      {active && (
                        <div className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-neon-green animate-pulse" style={{ boxShadow: "0 0 6px #00ffcc" }} />
                      )}
                    </div>
                  );
                })}
              </div>
            </SectionCard>

            <SectionCard>
              <SectionTitle icon="Columns" label="Колонки — распределение выборов" />
              <div className="flex items-end gap-3 h-40">
                {COLUMNS.map((c) => {
                  const active = c === activeColumn;
                  const h = Math.max(8, Math.round((colCounts[c] / maxColCount) * 120));
                  return (
                    <div key={c} className="flex-1 flex flex-col items-center gap-2">
                      <span className={`font-mono text-xs ${active ? "text-neon-green" : "text-white/30"}`}>{colCounts[c]}</span>
                      <div className="w-full flex flex-col justify-end" style={{ height: 120 }}>
                        <div
                          className="w-full rounded-t-md transition-all duration-700"
                          style={{
                            height: h,
                            background: active
                              ? "linear-gradient(180deg, #00ffcc, #00ffcc55)"
                              : "linear-gradient(180deg, rgba(168,85,247,0.6), rgba(168,85,247,0.15))",
                            boxShadow: active ? "0 0 16px rgba(0,255,204,0.5)" : "none",
                          }}
                        />
                      </div>
                      <span className={`font-display text-sm tracking-widest ${active ? "text-neon-green" : "text-white/50"}`}>{c}</span>
                    </div>
                  );
                })}
              </div>
            </SectionCard>

            <SectionCard>
              <SectionTitle icon="Radio" label="Поток событий (live)" />
              <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                {[...events].reverse().slice(0, 12).map((e, i) => (
                  <div
                    key={e.id}
                    className={`flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-mono transition-all duration-300 ${
                      i === 0 ? "bg-neon-green/10 border border-neon-green/20" : "bg-white/3 border border-transparent"
                    }`}
                  >
                    <span className="text-white/25 w-8 text-right flex-shrink-0">#{e.id}</span>
                    <span className={`font-bold w-10 ${i === 0 ? "text-neon-green" : "text-neon-blue/80"}`}>{e.reactor}</span>
                    <span className={`w-6 text-center font-bold ${i === 0 ? "text-neon-purple" : "text-neon-purple/60"}`}>{e.column}</span>
                    <span className="text-white/40 flex-1">{fmt(e.timestamp)}</span>
                    {e.interval !== null && <span className="text-white/25">+{e.interval}с</span>}
                    {i === 0 && <span className="text-neon-green animate-blink">▌</span>}
                  </div>
                ))}
              </div>
            </SectionCard>
          </div>
        )}

        {/* ── ИСТОРИЯ ── */}
        {tab === "history" && (
          <div className="space-y-5 animate-fade-in-up">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: "Всего событий", value: totalEvents, icon: "List", color: "text-neon-green" },
                { label: "Ср. интервал", value: `${avgInterval}с`, icon: "Timer", color: "text-neon-blue" },
                { label: "Мин. интервал", value: intervals.length ? `${Math.min(...intervals)}с` : "—", icon: "ArrowDown", color: "text-neon-purple" },
                { label: "Макс. интервал", value: intervals.length ? `${Math.max(...intervals)}с` : "—", icon: "ArrowUp", color: "text-neon-orange" },
              ].map((s) => (
                <SectionCard key={s.label} className="text-center">
                  <Icon name={s.icon} size={18} className={`${s.color} mx-auto mb-2`} />
                  <p className={`font-display text-2xl ${s.color}`}>{s.value}</p>
                  <p className="text-white/40 text-xs mt-1">{s.label}</p>
                </SectionCard>
              ))}
            </div>

            <SectionCard>
              <SectionTitle icon="Clock" label="Таблица событий" />
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="border-b border-white/10">
                      {["ID", "РЕАКТОР", "КОЛОНКА", "ВРЕМЯ", "ИНТЕРВАЛ"].map((h) => (
                        <th key={h} className="text-left py-2 px-3 text-white/30 font-normal tracking-widest">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[...events].reverse().map((e, i) => (
                      <tr key={e.id} className={`border-b border-white/5 transition-colors ${i === 0 ? "bg-neon-green/5" : "hover:bg-white/3"}`}>
                        <td className="py-2.5 px-3 text-white/25">#{e.id}</td>
                        <td className="py-2.5 px-3 text-neon-blue font-bold">{e.reactor}</td>
                        <td className="py-2.5 px-3">
                          <span className="inline-flex items-center justify-center w-7 h-7 rounded bg-neon-purple/15 border border-neon-purple/25 text-neon-purple font-bold">{e.column}</span>
                        </td>
                        <td className="py-2.5 px-3 text-white/60">{fmt(e.timestamp)}</td>
                        <td className="py-2.5 px-3 text-white/40">
                          {e.interval !== null
                            ? <span className={e.interval < avgInterval ? "text-neon-green" : "text-neon-orange"}>+{e.interval}с</span>
                            : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </SectionCard>
          </div>
        )}

        {/* ── ПРЕДСКАЗАНИЕ ── */}
        {tab === "prediction" && (
          <div className="space-y-5 animate-fade-in-up">
            <div className="grid sm:grid-cols-2 gap-5">
              <SectionCard className="scan-line">
                <SectionTitle icon="Zap" label="Следующее событие" />
                <div className="flex items-center gap-6 mt-2">
                  <div className="text-center">
                    <p className="text-white/30 text-xs font-mono mb-2 tracking-widest">РЕАКТОР</p>
                    <div className="w-20 h-20 rounded-xl bg-neon-green/10 border border-neon-green/30 flex items-center justify-center" style={{ boxShadow: "0 0 30px rgba(0,255,204,0.15)" }}>
                      <span className="font-display text-2xl text-neon-green">{predictedReactor}</span>
                    </div>
                  </div>
                  <Icon name="ArrowRight" size={24} className="text-white/20" />
                  <div className="text-center">
                    <p className="text-white/30 text-xs font-mono mb-2 tracking-widest">КОЛОНКА</p>
                    <div className="w-20 h-20 rounded-xl bg-neon-purple/10 border border-neon-purple/30 flex items-center justify-center" style={{ boxShadow: "0 0 30px rgba(168,85,247,0.15)" }}>
                      <span className="font-display text-4xl text-neon-purple">{predictedCol}</span>
                    </div>
                  </div>
                </div>
              </SectionCard>

              <SectionCard>
                <SectionTitle icon="Target" label="Уверенность модели" />
                <div className="mt-2">
                  <div className="flex items-end justify-between mb-3">
                    <span className="font-display text-5xl text-neon-green" style={{ textShadow: "0 0 30px rgba(0,255,204,0.4)" }}>
                      {Math.round(confidence * 100)}%
                    </span>
                    <span className="font-mono text-xs text-white/30">LSTM-v3</span>
                  </div>
                  <div className="h-3 rounded-full bg-white/5 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${confidence * 100}%`, background: "linear-gradient(90deg, #00ffcc, #a855f7)", boxShadow: "0 0 10px rgba(0,255,204,0.5)" }}
                    />
                  </div>
                  <div className="flex justify-between mt-1.5">
                    <span className="font-mono text-xs text-white/20">0%</span>
                    <span className="font-mono text-xs text-white/20">100%</span>
                  </div>
                </div>
              </SectionCard>
            </div>

            <SectionCard>
              <SectionTitle icon="PieChart" label="Вероятность по колонкам" />
              <div className="grid grid-cols-6 gap-2">
                {COLUMNS.map((c) => {
                  const isTarget = c === predictedCol;
                  const normalized = isTarget ? confidence : (1 - confidence) / 5;
                  return (
                    <div key={c} className="text-center">
                      <div className="h-24 flex flex-col justify-end mb-2">
                        <div
                          className="w-full rounded-t-md transition-all duration-700"
                          style={{
                            height: `${Math.max(6, normalized * 96)}px`,
                            background: isTarget ? "linear-gradient(180deg, #00ffcc, #00ffcc33)" : "linear-gradient(180deg, rgba(168,85,247,0.5), rgba(168,85,247,0.1))",
                            boxShadow: isTarget ? "0 0 20px rgba(0,255,204,0.4)" : "none",
                          }}
                        />
                      </div>
                      <p className={`font-display text-sm tracking-widest ${isTarget ? "text-neon-green" : "text-white/40"}`}>{c}</p>
                      <p className={`font-mono text-xs mt-0.5 ${isTarget ? "text-neon-green" : "text-white/25"}`}>{Math.round(normalized * 100)}%</p>
                    </div>
                  );
                })}
              </div>
            </SectionCard>

            <SectionCard>
              <SectionTitle icon="GitBranch" label="Последовательность (контекст предсказания)" />
              <div className="flex items-center gap-2 flex-wrap">
                {events.slice(-8).map((e, i, arr) => (
                  <div key={e.id} className="flex items-center gap-2">
                    <div className={`flex flex-col items-center gap-1 px-3 py-2 rounded-lg border transition-all ${i === arr.length - 1 ? "bg-neon-green/10 border-neon-green/30" : "bg-white/3 border-white/10"}`}>
                      <span className={`font-mono text-xs ${i === arr.length - 1 ? "text-neon-blue" : "text-white/30"}`}>{e.reactor}</span>
                      <span className={`font-display text-lg ${i === arr.length - 1 ? "text-neon-green" : "text-white/50"}`}>{e.column}</span>
                    </div>
                    {i < arr.length - 1 && <Icon name="ChevronRight" size={12} className="text-white/20" />}
                  </div>
                ))}
                <Icon name="ChevronRight" size={12} className="text-neon-green/50" />
                <div className="flex flex-col items-center gap-1 px-3 py-2 rounded-lg border border-neon-green/50 bg-neon-green/5" style={{ borderStyle: "dashed" }}>
                  <span className="font-mono text-xs text-neon-purple/70">{predictedReactor}</span>
                  <span className="font-display text-lg text-neon-green animate-pulse">{predictedCol}</span>
                </div>
              </div>
            </SectionCard>
          </div>
        )}

        {/* ── СТАТИСТИКА ── */}
        {tab === "stats" && (
          <div className="space-y-5 animate-fade-in-up">
            <div className="grid sm:grid-cols-2 gap-5">
              <SectionCard>
                <SectionTitle icon="BarChart2" label="Частота по колонкам" />
                <div className="space-y-2.5">
                  {COLUMNS.map((c) => {
                    const pct = Math.round((colCounts[c] / totalEvents) * 100);
                    return (
                      <div key={c} className="flex items-center gap-3">
                        <span className="font-display text-sm text-white/60 w-5">{c}</span>
                        <div className="flex-1 h-6 bg-white/5 rounded-md overflow-hidden">
                          <div
                            className="h-full rounded-md transition-all duration-700 flex items-center px-2"
                            style={{
                              width: `${Math.max(5, pct)}%`,
                              background: c === activeColumn ? "linear-gradient(90deg, #00ffcc, #00ffcc88)" : "linear-gradient(90deg, rgba(168,85,247,0.6), rgba(168,85,247,0.3))",
                            }}
                          >
                            {pct > 10 && <span className="font-mono text-xs text-[#080d14] font-bold">{pct}%</span>}
                          </div>
                        </div>
                        <span className="font-mono text-xs text-white/30 w-8 text-right">{colCounts[c]}</span>
                      </div>
                    );
                  })}
                </div>
              </SectionCard>

              <SectionCard>
                <SectionTitle icon="Activity" label="Активность реакторов %" />
                <div className="space-y-2.5">
                  {REACTORS.map((r) => {
                    const pct = Math.round((reactorCounts[r] / totalEvents) * 100);
                    return (
                      <div key={r} className="flex items-center gap-3">
                        <span className="font-mono text-xs text-white/40 w-10">{r}</span>
                        <div className="flex-1 h-5 bg-white/5 rounded overflow-hidden">
                          <div
                            className="h-full rounded transition-all duration-700"
                            style={{
                              width: `${Math.max(4, pct)}%`,
                              background: r === activeReactor ? "linear-gradient(90deg, #00ffcc, #38bdf8)" : "linear-gradient(90deg, rgba(56,189,248,0.5), rgba(56,189,248,0.2))",
                              boxShadow: r === activeReactor ? "0 0 8px rgba(0,255,204,0.4)" : "none",
                            }}
                          />
                        </div>
                        <span className="font-mono text-xs text-white/30 w-8 text-right">{pct}%</span>
                      </div>
                    );
                  })}
                </div>
              </SectionCard>
            </div>

            <SectionCard>
              <SectionTitle icon="GitMerge" label="Найденные паттерны последовательностей" />
              <div className="grid sm:grid-cols-2 gap-3">
                {PATTERNS.map((p, i) => (
                  <div key={i} className="glass-card-purple rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <span className="font-mono text-xs text-white/40">{p.label}</span>
                      <span className="font-mono text-xs text-neon-purple">{Math.round(p.confidence * 100)}% conf</span>
                    </div>
                    <div className="flex items-center gap-1.5 mb-3">
                      {p.seq.map((s, j) => (
                        <div key={j} className="flex items-center gap-1.5">
                          <span className="w-7 h-7 flex items-center justify-center rounded bg-neon-purple/15 border border-neon-purple/25 font-display text-sm text-neon-purple">{s}</span>
                          {j < p.seq.length - 1 && <Icon name="ArrowRight" size={10} className="text-white/20" />}
                        </div>
                      ))}
                    </div>
                    <div className="h-1.5 rounded-full bg-white/5">
                      <div className="h-full rounded-full" style={{ width: `${p.confidence * 100}%`, background: "linear-gradient(90deg, #a855f7, #38bdf8)" }} />
                    </div>
                    <p className="font-mono text-xs text-white/25 mt-2">встречается {p.count} раз</p>
                  </div>
                ))}
              </div>
            </SectionCard>

            <SectionCard>
              <SectionTitle icon="Flame" label="Тепловая карта активности (последние 50 событий)" />
              <div className="flex flex-wrap gap-1">
                {events.slice(-50).map((e, i) => {
                  const intensity = (i + 1) / 50;
                  return (
                    <div
                      key={e.id}
                      title={`${e.reactor} / ${e.column} — ${fmt(e.timestamp)}`}
                      className="w-6 h-6 rounded-sm cursor-default transition-all hover:scale-110"
                      style={{
                        background: `rgba(0, 255, 204, ${0.08 + intensity * 0.55})`,
                        border: `1px solid rgba(0, 255, 204, ${0.08 + intensity * 0.28})`,
                      }}
                    />
                  );
                })}
              </div>
              <p className="font-mono text-xs text-white/20 mt-3">Наведи на ячейку для деталей · слева → давнее, справа → новее</p>
            </SectionCard>
          </div>
        )}

        {/* ── МОДЕЛЬ ML ── */}
        {tab === "model" && (
          <div className="space-y-5 animate-fade-in-up">
            <div className="grid sm:grid-cols-3 gap-4">
              {[
                { label: "Тип модели", value: "LSTM", sub: "Long Short-Term Memory", color: "text-neon-green", icon: "Cpu" },
                { label: "Точность", value: "87.4%", sub: "на тестовой выборке", color: "text-neon-blue", icon: "Target" },
                { label: "Обновление", value: "real-time", sub: "скользящее окно 50 evt", color: "text-neon-purple", icon: "RefreshCw" },
              ].map((s) => (
                <SectionCard key={s.label} className="text-center">
                  <Icon name={s.icon} size={22} className={`${s.color} mx-auto mb-3`} />
                  <p className={`font-display text-2xl ${s.color} mb-1`}>{s.value}</p>
                  <p className="text-white/40 text-xs font-mono">{s.label}</p>
                  <p className="text-white/25 text-xs mt-1">{s.sub}</p>
                </SectionCard>
              ))}
            </div>

            <div className="grid sm:grid-cols-2 gap-5">
              <SectionCard>
                <SectionTitle icon="Settings" label="Параметры обучения" />
                <div className="space-y-3 font-mono text-sm">
                  {[
                    ["Длина последовательности", "8 событий"],
                    ["Размер скрытого слоя", "128 нейронов"],
                    ["Dropout", "0.2"],
                    ["Learning rate", "0.001"],
                    ["Batch size", "32"],
                    ["Эпох обучения", "200"],
                    ["Оптимизатор", "Adam"],
                    ["Loss function", "Cross-entropy"],
                  ].map(([k, v]) => (
                    <div key={k} className="flex items-center justify-between border-b border-white/5 pb-2">
                      <span className="text-white/40 text-xs">{k}</span>
                      <span className="text-neon-green text-xs font-bold">{v}</span>
                    </div>
                  ))}
                </div>
              </SectionCard>

              <div className="space-y-4">
                <SectionCard>
                  <SectionTitle icon="TrendingUp" label="Метрики качества" />
                  <div className="space-y-3">
                    {[
                      { label: "Top-1 Accuracy", val: 0.874, color: "#00ffcc" },
                      { label: "Top-3 Accuracy", val: 0.961, color: "#38bdf8" },
                      { label: "F1-score", val: 0.851, color: "#a855f7" },
                      { label: "AUC-ROC", val: 0.923, color: "#fb923c" },
                    ].map((m) => (
                      <div key={m.label}>
                        <div className="flex justify-between mb-1">
                          <span className="font-mono text-xs text-white/40">{m.label}</span>
                          <span className="font-mono text-xs font-bold" style={{ color: m.color }}>{Math.round(m.val * 100)}%</span>
                        </div>
                        <div className="h-1.5 rounded-full bg-white/5">
                          <div className="h-full rounded-full" style={{ width: `${m.val * 100}%`, background: m.color, boxShadow: `0 0 6px ${m.color}66` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </SectionCard>

                <SectionCard>
                  <SectionTitle icon="Lightbulb" label="Найденные закономерности" />
                  <ul className="space-y-2">
                    {[
                      "Циклическая ротация A→B→C встречается в 23% цепочек",
                      "R-03 имеет повышенную активность в 18:00–22:00",
                      "Колонка D часто следует за F с вероятностью 41%",
                      "Медианный интервал между событиями: 4.2 сек",
                    ].map((insight, i) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-white/50 font-mono">
                        <span className="text-neon-green mt-0.5 flex-shrink-0">◆</span>
                        <span>{insight}</span>
                      </li>
                    ))}
                  </ul>
                </SectionCard>
              </div>
            </div>

            <SectionCard>
              <SectionTitle icon="Network" label="Архитектура нейросети" />
              <div className="flex items-center gap-3 overflow-x-auto py-2">
                {[
                  { label: "INPUT", sub: "seq × 2 features", color: "#38bdf8" },
                  { label: "EMBED", sub: "dim=32", color: "#38bdf8" },
                  { label: "LSTM", sub: "128 units", color: "#00ffcc" },
                  { label: "LSTM", sub: "64 units", color: "#00ffcc" },
                  { label: "DROPOUT", sub: "p=0.2", color: "#facc15" },
                  { label: "DENSE", sub: "32 units, ReLU", color: "#a855f7" },
                  { label: "OUTPUT", sub: `${COLUMNS.length} classes`, color: "#f43f5e" },
                ].map((layer, i, arr) => (
                  <div key={i} className="flex items-center gap-2 flex-shrink-0">
                    <div className="text-center px-3 py-2 rounded-lg border" style={{ borderColor: `${layer.color}44`, background: `${layer.color}0d` }}>
                      <p className="font-display text-xs tracking-widest" style={{ color: layer.color }}>{layer.label}</p>
                      <p className="font-mono text-xs text-white/30 mt-0.5 whitespace-nowrap">{layer.sub}</p>
                    </div>
                    {i < arr.length - 1 && <Icon name="ArrowRight" size={14} className="text-white/20" />}
                  </div>
                ))}
              </div>
            </SectionCard>
          </div>
        )}
      </main>
    </div>
  );
}