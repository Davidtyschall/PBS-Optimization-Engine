/**
 * API Client for PBS Optimizer Backend
 * =====================================
 * 
 * This file handles all communication between the Next.js frontend
 * and the FastAPI backend running on localhost:8000.
 * 
 * WHY THIS EXISTS:
 * - Centralizes all API calls in one place
 * - Defines TypeScript types for request/response shapes
 * - Handles errors consistently
 * - Makes it easy to change the API URL later (e.g., for production)
 */

// The base URL for your FastAPI backend
// In development: http://localhost:8000
// In production: You'd change this to your deployed API URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================
// These match exactly what your FastAPI backend sends/receives.
// TypeScript uses these to catch errors at compile time.

/**
 * Response from POST /api/preferences/parse
 * 
 * When you send natural language like "Fridays off, Hawaii trips",
 * the backend parses it into structured data.
 */
export interface ParsePreferencesResponse {
  success: boolean;
  confidence: "high" | "medium" | "low";
  preferences: Record<string, unknown> | null;  // The structured preferences object
  warnings: string[];
  errors: string[];
  clarification_needed: string | null;
  tokens_used: number;
  cached: boolean;
}

/**
 * A single category's score breakdown (e.g., "days_off", "layovers")
 * 
 * This shows HOW a sequence scored in each category.
 */
export interface CategoryScore {
  score: number;       // Raw score (0-100)
  max_score: number;   // Maximum possible
  weight: number;      // How important this category is (0-1)
  weighted_score: number;  // score * weight
  factors: string[];   // Human-readable explanations like "+15 Fridays off"
}

/**
 * A scored sequence with all its details
 * 
 * This is one row in your results table.
 */
export interface ScoredSequence {
  sequence_id: number;
  seq_num: number;           // The sequence number pilots see (e.g., 224)
  rank: number;              // Position in ranked list (1 = best)
  final_score: number;       // Overall score (0-100)
  disqualified: boolean;     // If true, this sequence is filtered out
  disqualification_reason: string | null;
  breakdown: Record<string, CategoryScore>;  // Score by category
  explanation: string;       // Human-readable summary
  
  // Sequence details
  calendar_start_dates: number[];  // Days of month this sequence starts
  layover_cities: string[];        // Where you overnight
  pairing_length: number;          // How many days
  total_credit_minutes: number;    // Pay value
  total_tafb_minutes: number;      // Time away from base
}

/**
 * Response from POST /api/sequences/score
 * 
 * The full scoring result with all ranked sequences.
 */
export interface ScoreSequencesResponse {
  success: boolean;
  ranked_sequences: ScoredSequence[];
  total_sequences: number;
  disqualified_count: number;
  preferences_summary: string;
  error: string | null;
}

/**
 * A bid packet (the monthly PDF you upload)
 */
export interface BidPacket {
  id: number;
  airline: string;
  base: string;
  equipment: string;
  division: string;
  bid_month: string;
  filename: string;
  page_count: number;
  pages_parsed: number;
  parse_status: string;
  parse_confidence: string | null;
}

/**
 * Response from GET /api/packets
 */
export interface ListPacketsResponse {
  success: boolean;
  packets: BidPacket[];
  total: number;
  error: string | null;
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

/**
 * Parse natural language preferences into structured format.
 * 
 * EXAMPLE:
 *   Input:  "I want Fridays off, Hawaii layovers, no redeyes"
 *   Output: { days_off: { days_off: ["Friday"] }, ... }
 * 
 * @param text - Natural language preference text
 * @param bidMonth - Month in YYYY-MM format (default: "2026-01")
 */
export async function parsePreferences(
  text: string,
  bidMonth: string = "2026-01"
): Promise<ParsePreferencesResponse> {
  const response = await fetch(`${API_BASE_URL}/api/preferences/parse`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text,
      bid_month: bidMonth,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to parse preferences: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Score sequences against pilot preferences.
 * 
 * This is the core ranking function. It takes your structured preferences
 * and scores every sequence in the specified bid packet.
 * 
 * @param preferences - Structured preferences object (from parsePreferences)
 * @param packetId - ID of the bid packet to score
 * @param bidYear - Year (default: 2026)
 * @param bidMonth - Month number 1-12 (default: 1)
 */
export async function scoreSequences(
  preferences: Record<string, unknown>,
  packetId: number,
  bidYear: number = 2026,
  bidMonth: number = 1
): Promise<ScoreSequencesResponse> {
  const response = await fetch(`${API_BASE_URL}/api/sequences/score`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      preferences,
      packet_id: packetId,
      bid_year: bidYear,
      bid_month: bidMonth,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to score sequences: ${response.statusText}`);
  }

  return response.json();
}

/**
 * List all bid packets in the database.
 * 
 * In the future, this would be filtered by user.
 * For now, it returns all packets.
 */
export async function listPackets(): Promise<ListPacketsResponse> {
  const response = await fetch(`${API_BASE_URL}/api/packets`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to list packets: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Health check - verify the API is running
 */
export async function healthCheck(): Promise<{ status: string; version: string }> {
  const response = await fetch(`${API_BASE_URL}/health`);
  
  if (!response.ok) {
    throw new Error("API is not responding");
  }

  return response.json();
}
