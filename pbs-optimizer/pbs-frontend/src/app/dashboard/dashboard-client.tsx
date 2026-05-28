"use client";

import { useState, useCallback, useEffect } from "react";
import { User } from "@supabase/supabase-js";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Plane,
  Upload,
  FileText,
  Sparkles,
  ArrowRight,
  Loader2,
  CheckCircle2,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  X,
  AlertTriangle,
} from "lucide-react";
import SignOutButton from "./sign-out-button";

// Import our API client
import {
  parsePreferences,
  scoreSequences,
  listPackets,
  healthCheck,
  type ScoredSequence,
  type BidPacket,
} from "@/lib/api";

// Processing state type
interface ProcessingState {
  step: "idle" | "checking" | "parsing" | "scoring" | "complete" | "error";
  message: string;
  progress: number;
}

export default function DashboardClient({ user }: { user: User }) {
  // ==========================================================================
  // STATE
  // ==========================================================================
  
  // File upload (not used yet - for future PDF upload feature)
  const [file, setFile] = useState<File | null>(null);
  
  // User's natural language preferences
  const [preferences, setPreferences] = useState("");
  
  // Ranked sequences from the API
  const [sequences, setSequences] = useState<ScoredSequence[]>([]);
  
  // Which sequence card is expanded (to show breakdown)
  const [expandedSequence, setExpandedSequence] = useState<number | null>(null);
  
  // Processing state for the progress indicator
  const [processing, setProcessing] = useState<ProcessingState>({
    step: "idle",
    message: "",
    progress: 0,
  });
  
  // Available bid packets from the database
  const [packets, setPackets] = useState<BidPacket[]>([]);
  
  // Currently selected packet ID
  const [selectedPacketId, setSelectedPacketId] = useState<number | null>(null);
  
  // API connection status
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  
  // Error message to display
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // ==========================================================================
  // EFFECTS
  // ==========================================================================
  
  /**
   * On component mount:
   * 1. Check if the FastAPI backend is running
   * 2. Load available bid packets
   * 
   * WHY: We need to know if the API is available before the user tries to use it.
   */
  useEffect(() => {
    async function initialize() {
      try {
        // Step 1: Health check
        await healthCheck();
        setApiStatus("online");
        
        // Step 2: Load packets
        const packetsResponse = await listPackets();
        if (packetsResponse.success && packetsResponse.packets.length > 0) {
          setPackets(packetsResponse.packets);
          // Auto-select the first packet
          setSelectedPacketId(packetsResponse.packets[0].id);
        }
      } catch (error) {
        console.error("API initialization failed:", error);
        setApiStatus("offline");
      }
    }
    
    initialize();
  }, []);

  // ==========================================================================
  // HANDLERS
  // ==========================================================================

  // File drop handler (for future PDF upload feature)
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile?.type === "application/pdf") {
      setFile(droppedFile);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile?.type === "application/pdf") {
      setFile(selectedFile);
    }
  };

  const removeFile = () => {
    setFile(null);
    setSequences([]);
    setProcessing({ step: "idle", message: "", progress: 0 });
  };

  /**
   * THE MAIN FUNCTION: Process preferences and score sequences
   * 
   * This is what happens when the user clicks "Analyze & Rank Sequences":
   * 
   * 1. PARSE: Send natural language to /api/preferences/parse
   *    - Input:  "I want Fridays off, Hawaii trips, no redeyes"
   *    - Output: { days_off: { days_off: ["Friday"] }, ... }
   * 
   * 2. SCORE: Send structured preferences to /api/sequences/score
   *    - Input:  { preferences: {...}, packet_id: 1 }
   *    - Output: { ranked_sequences: [...] }
   * 
   * 3. DISPLAY: Update the UI with ranked results
   */
  const handleProcess = async () => {
    // Validation
    if (!preferences.trim()) {
      setErrorMessage("Please enter your preferences first.");
      return;
    }
    
    if (!selectedPacketId) {
      setErrorMessage("No bid packet available. Please contact support.");
      return;
    }

    // Reset state
    setErrorMessage(null);
    setSequences([]);

    try {
      // =====================================================================
      // STEP 1: PARSE PREFERENCES
      // =====================================================================
      // Convert natural language like "Fridays off, Hawaii trips" 
      // into structured data the scoring engine can use.
      
      setProcessing({ 
        step: "parsing", 
        message: "Understanding your preferences...", 
        progress: 30 
      });

      const parseResult = await parsePreferences(preferences.trim());
      
      // Check if parsing succeeded
      if (!parseResult.success || !parseResult.preferences) {
        throw new Error(
          parseResult.errors?.join(", ") || 
          parseResult.clarification_needed || 
          "Could not understand your preferences. Try being more specific."
        );
      }

      // Show confidence level
      const confidenceMsg = parseResult.confidence === "high" 
        ? "Understood your preferences clearly!" 
        : parseResult.confidence === "medium"
        ? "Understood most of your preferences..."
        : "Interpreted your preferences (some ambiguity)...";
      
      setProcessing({ 
        step: "parsing", 
        message: confidenceMsg, 
        progress: 50 
      });

      // Brief pause so user can see the message
      await sleep(500);

      // =====================================================================
      // STEP 2: SCORE SEQUENCES
      // =====================================================================
      // Now we have structured preferences. Send them to the scoring engine
      // along with the packet ID to score all sequences.

      setProcessing({ 
        step: "scoring", 
        message: "Scoring sequences against your preferences...", 
        progress: 70 
      });

      const scoreResult = await scoreSequences(
        parseResult.preferences,
        selectedPacketId
      );

      // Check if scoring succeeded
      if (!scoreResult.success) {
        throw new Error(scoreResult.error || "Failed to score sequences.");
      }

      // =====================================================================
      // STEP 3: DISPLAY RESULTS
      // =====================================================================
      
      setProcessing({ 
        step: "complete", 
        message: `Ranked ${scoreResult.total_sequences} sequences!`, 
        progress: 100 
      });

      // Store the ranked sequences for display
      setSequences(scoreResult.ranked_sequences);

    } catch (error) {
      console.error("Processing failed:", error);
      setProcessing({ 
        step: "error", 
        message: "Something went wrong", 
        progress: 0 
      });
      setErrorMessage(
        error instanceof Error ? error.message : "An unexpected error occurred."
      );
    }
  };

  // Helper function for artificial delays (UX purposes)
  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

  // ==========================================================================
  // HELPER FUNCTIONS FOR DISPLAY
  // ==========================================================================

  // Score color based on value
  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-400 bg-green-500/10";
    if (score >= 60) return "text-amber bg-amber/10";
    return "text-slate-warm bg-white/5";
  };

  // Progress bar width
  const getScoreBarWidth = (score: number) => `${Math.min(score, 100)}%`;

  // Convert minutes to hours:minutes format
  const formatMinutes = (minutes: number): string => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}:${mins.toString().padStart(2, "0")}`;
  };

  // Extract explanation factors from breakdown
  const getExplanationFactors = (breakdown: Record<string, { factors: string[] }>): string[] => {
    const factors: string[] = [];
    for (const category of Object.values(breakdown)) {
      factors.push(...category.factors);
    }
    // Sort: positives first, then negatives
    return factors.sort((a, b) => {
      if (a.startsWith("+") && b.startsWith("-")) return -1;
      if (a.startsWith("-") && b.startsWith("+")) return 1;
      return 0;
    });
  };

  return (
    <div className="min-h-screen bg-navy">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-white/5 bg-navy/80 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber/10">
              <Plane className="h-4 w-4 text-amber" />
            </div>
            <span className="text-lg font-semibold tracking-tight text-white">
              Bid<span className="text-amber">Line</span>
            </span>
          </div>

          <div className="flex items-center gap-4">
            {/* API Status Indicator */}
            <div className="flex items-center gap-2">
              <div
                className={`h-2 w-2 rounded-full ${
                  apiStatus === "online"
                    ? "bg-green-400"
                    : apiStatus === "offline"
                    ? "bg-red-400"
                    : "bg-yellow-400 animate-pulse"
                }`}
              />
              <span className="hidden text-xs text-slate-warm/60 sm:block">
                {apiStatus === "online" ? "API Connected" : apiStatus === "offline" ? "API Offline" : "Connecting..."}
              </span>
            </div>
            <span className="hidden text-sm text-slate-warm/60 sm:block">{user.email}</span>
            <SignOutButton />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-6 py-8">
        {/* API Offline Warning */}
        {apiStatus === "offline" && (
          <div className="mb-6 flex items-center gap-3 rounded-lg border border-red-500/20 bg-red-500/10 p-4">
            <AlertTriangle className="h-5 w-5 text-red-400" />
            <div>
              <p className="text-sm font-medium text-red-400">API Connection Failed</p>
              <p className="text-xs text-red-400/80">
                Make sure your FastAPI backend is running on localhost:8000
              </p>
            </div>
          </div>
        )}

        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-white">Optimize Your Bid</h1>
          <p className="mt-1 text-slate-warm/60">
            {packets.length > 0
              ? `Analyzing ${packets[0].base} ${packets[0].equipment} ${packets[0].division} - ${packets[0].bid_month}`
              : "Enter your preferences and we'll rank your sequences."}
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          {/* Left Column: Upload + Preferences */}
          <div className="space-y-6">
            {/* Upload Section (Placeholder for now) */}
            <section className="rounded-xl border border-white/5 bg-navy-light p-6">
              <div className="mb-4 flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-amber/10">
                  <Upload className="h-4 w-4 text-amber" />
                </div>
                <div>
                  <h2 className="font-semibold text-white">Bid Packet</h2>
                  <p className="text-xs text-slate-warm/60">
                    {selectedPacketId
                      ? `Using packet #${selectedPacketId} from database`
                      : "No packet loaded"}
                  </p>
                </div>
              </div>

              {/* Show current packet info or upload placeholder */}
              {packets.length > 0 ? (
                <div className="flex items-center gap-3 rounded-lg border border-white/10 bg-navy/40 p-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-green-500/10">
                    <CheckCircle2 className="h-5 w-5 text-green-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="truncate text-sm font-medium text-white">
                      {packets[0].filename}
                    </p>
                    <p className="text-xs text-slate-warm/60">
                      {packets[0].pages_parsed} of {packets[0].page_count} pages parsed •{" "}
                      {packets[0].parse_status}
                    </p>
                  </div>
                </div>
              ) : (
                <div
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-white/10 bg-navy/40 p-8 transition-colors hover:border-amber/30 hover:bg-navy/60"
                >
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileSelect}
                    className="absolute inset-0 cursor-pointer opacity-0"
                    id="file-upload"
                  />
                  <label htmlFor="file-upload" className="cursor-pointer text-center">
                    <FileText className="mx-auto mb-3 h-10 w-10 text-slate-warm/40" />
                    <p className="text-sm font-medium text-white">Drop your bid packet here</p>
                    <p className="mt-1 text-xs text-slate-warm/60">or click to browse</p>
                  </label>
                </div>
              )}
            </section>

            {/* Preferences Section */}
            <section className="rounded-xl border border-white/5 bg-navy-light p-6">
              <div className="mb-4 flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-amber/10">
                  <Sparkles className="h-4 w-4 text-amber" />
                </div>
                <div>
                  <h2 className="font-semibold text-white">Your Preferences</h2>
                  <p className="text-xs text-slate-warm/60">Tell us what matters to you in plain English</p>
                </div>
              </div>

              <Textarea
                value={preferences}
                onChange={(e) => setPreferences(e.target.value)}
                placeholder="Example: I want Fridays off, prefer Hawaii layovers, no redeyes, and maximize my pay..."
                className="min-h-[120px] resize-none border-white/10 bg-navy/40 text-white placeholder:text-slate-warm/40 focus:border-amber/50 focus:ring-amber/20"
              />

              {/* Quick-add tags */}
              <div className="mt-3 flex flex-wrap gap-2">
                {["Fridays off", "No redeyes", "Hawaii layovers", "Max pay"].map((tag) => (
                  <button
                    key={tag}
                    onClick={() =>
                      setPreferences((prev) => (prev ? `${prev}, ${tag.toLowerCase()}` : tag.toLowerCase()))
                    }
                    className="rounded-full border border-white/10 bg-navy/40 px-3 py-1 text-xs text-slate-warm/80 transition-colors hover:border-amber/30 hover:text-amber"
                  >
                    + {tag}
                  </button>
                ))}
              </div>
            </section>

            {/* Error Message */}
            {errorMessage && (
              <div className="flex items-center gap-3 rounded-lg border border-red-500/20 bg-red-500/10 p-4">
                <AlertCircle className="h-5 w-5 flex-shrink-0 text-red-400" />
                <p className="text-sm text-red-400">{errorMessage}</p>
                <button
                  onClick={() => setErrorMessage(null)}
                  className="ml-auto text-red-400/60 hover:text-red-400"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            )}

            {/* Process Button */}
            <Button
              onClick={handleProcess}
              disabled={
                !preferences.trim() ||
                !selectedPacketId ||
                apiStatus !== "online" ||
                processing.step === "parsing" ||
                processing.step === "scoring"
              }
              className="w-full bg-amber py-6 text-base font-semibold text-navy hover:bg-amber-light disabled:opacity-50"
            >
              {processing.step !== "idle" && processing.step !== "complete" && processing.step !== "error" ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  {processing.message}
                </>
              ) : (
                <>
                  Analyze & Rank Sequences
                  <ArrowRight className="ml-2 h-5 w-5" />
                </>
              )}
            </Button>

            {/* Processing Progress */}
            {processing.step !== "idle" && processing.step !== "error" && (
              <div className="rounded-lg border border-white/5 bg-navy/40 p-4">
                <div className="mb-2 flex items-center justify-between text-sm">
                  <span className="text-slate-warm/80">{processing.message}</span>
                  <span className="text-amber">{processing.progress}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-white/5">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-amber to-amber-light transition-all duration-500"
                    style={{ width: `${processing.progress}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Results */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">Ranked Sequences</h2>
              {sequences.length > 0 && (
                <span className="text-sm text-slate-warm/60">{sequences.length} sequences</span>
              )}
            </div>

            {sequences.length === 0 ? (
              <div className="flex flex-col items-center justify-center rounded-xl border border-white/5 bg-navy-light p-12 text-center">
                <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-white/5">
                  <Sparkles className="h-7 w-7 text-slate-warm/40" />
                </div>
                <p className="text-sm text-slate-warm/60">
                  Enter your preferences and click &quot;Analyze&quot; to see ranked sequences.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {sequences.map((seq, index) => (
                  <div
                    key={seq.sequence_id}
                    className="rounded-xl border border-white/5 bg-navy-light transition-all hover:border-white/10"
                  >
                    {/* Sequence Header */}
                    <button
                      onClick={() =>
                        setExpandedSequence(
                          expandedSequence === seq.sequence_id ? null : seq.sequence_id
                        )
                      }
                      className="flex w-full items-center gap-4 p-4"
                    >
                      {/* Rank Badge */}
                      <div
                        className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg text-sm font-bold ${
                          index === 0 ? "bg-amber/10 text-amber" : "bg-white/5 text-slate-warm"
                        }`}
                      >
                        #{seq.rank}
                      </div>

                      {/* Sequence Info */}
                      <div className="flex-1 text-left">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-white">SEQ {seq.seq_num}</span>
                          <span
                            className={`rounded px-2 py-0.5 text-xs font-medium ${getScoreColor(seq.final_score)}`}
                          >
                            {seq.final_score.toFixed(1)}
                          </span>
                          {seq.disqualified && (
                            <span className="rounded bg-red-500/10 px-2 py-0.5 text-xs text-red-400">
                              DQ
                            </span>
                          )}
                        </div>
                        <p className="mt-0.5 text-xs text-slate-warm/60">
                          {seq.pairing_length} days • {formatMinutes(seq.total_credit_minutes)} credit •{" "}
                          {seq.layover_cities.length > 0 ? seq.layover_cities.join(", ") : "Day trips"}
                        </p>
                      </div>

                      {/* Score Bar */}
                      <div className="hidden w-24 sm:block">
                        <div className="h-2 overflow-hidden rounded-full bg-white/5">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-amber to-amber-light"
                            style={{ width: getScoreBarWidth(seq.final_score) }}
                          />
                        </div>
                      </div>

                      {/* Expand Icon */}
                      {expandedSequence === seq.sequence_id ? (
                        <ChevronUp className="h-5 w-5 text-slate-warm/60" />
                      ) : (
                        <ChevronDown className="h-5 w-5 text-slate-warm/60" />
                      )}
                    </button>

                    {/* Expanded Details */}
                    {expandedSequence === seq.sequence_id && (
                      <div className="border-t border-white/5 p-4">
                        {/* Disqualification reason */}
                        {seq.disqualified && seq.disqualification_reason && (
                          <div className="mb-3 rounded-lg border border-red-500/20 bg-red-500/10 p-3">
                            <p className="text-sm text-red-400">
                              <strong>Disqualified:</strong> {seq.disqualification_reason}
                            </p>
                          </div>
                        )}

                        {/* Explanation */}
                        <p className="mb-3 text-xs font-medium uppercase tracking-wide text-slate-warm/60">
                          Score Breakdown
                        </p>
                        <div className="space-y-2">
                          {getExplanationFactors(seq.breakdown).slice(0, 6).map((item, i) => (
                            <div key={i} className="flex items-start gap-2 text-sm">
                              {item.startsWith("+") ? (
                                <CheckCircle2 className="mt-0.5 h-4 w-4 flex-shrink-0 text-green-400" />
                              ) : (
                                <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-rose-400" />
                              )}
                              <span
                                className={
                                  item.startsWith("+") ? "text-green-400/90" : "text-rose-400/90"
                                }
                              >
                                {item}
                              </span>
                            </div>
                          ))}
                        </div>

                        {/* Calendar Days */}
                        <div className="mt-4">
                          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-warm/60">
                            Working Days
                          </p>
                          <div className="flex flex-wrap gap-1.5">
                            {seq.calendar_start_dates.map((day) => (
                              <span
                                key={day}
                                className="flex h-7 w-7 items-center justify-center rounded bg-white/5 text-xs font-medium text-slate-warm"
                              >
                                {day}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
