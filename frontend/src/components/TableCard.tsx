import { Clock, Users } from "lucide-react";

import type { TableStatus } from "../api/types";
import { formatDuration } from "../lib/format";

const PENDING_NOTE: Partial<Record<TableStatus["status"], string>> = {
  PENDING_OCCUPIED: "Someone arrived, confirming…",
  PENDING_AVAILABLE: "Guests left, confirming…",
};

/** One table: big AVAILABLE / OCCUPIED badge, people count and how long it has been occupied. */
export default function TableCard({ table, now }: { table: TableStatus; now: number }) {
  const occupiedFor =
    table.status === "OCCUPIED" && table.occupied_since !== null
      ? now - table.occupied_since
      : table.current_session_seconds; // frozen while waiting to become AVAILABLE
  const note = PENDING_NOTE[table.status];
  const badge = table.needs_cleaning
    ? { text: "NEEDS CLEANING", classes: "bg-orange-500" }
    : table.occupied
      ? { text: "OCCUPIED", classes: "bg-red-600" }
      : { text: "AVAILABLE", classes: "bg-emerald-600" };

  return (
    <div
      className={`rounded-lg border p-3 ${
        table.needs_cleaning
          ? "border-orange-200 bg-orange-50/60"
          : table.occupied
            ? "border-red-200 bg-red-50/60"
            : "border-emerald-200 bg-emerald-50/60"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <h3 className="truncate text-sm font-semibold text-slate-800" title={table.name}>
          {table.name}
        </h3>
        <span className="flex items-center gap-1 text-xs text-slate-500" title="People at the table">
          <Users className="h-3.5 w-3.5" />
          {table.people_count}
        </span>
      </div>
      <div
        className={`mt-2 rounded-md py-1.5 text-center text-sm font-bold tracking-wide text-white ${badge.classes}`}
      >
        {badge.text}
      </div>
      <p className="mt-2 flex h-4 items-center gap-1 text-xs text-slate-600">
        {table.occupied ? (
          <>
            <Clock className="h-3.5 w-3.5" />
            {formatDuration(occupiedFor)}
          </>
        ) : null}
        {table.unattended && <span className="ml-auto font-semibold text-red-700">No staff visit yet</span>}
        {note && <span className="ml-auto italic text-amber-700">{note}</span>}
      </p>
    </div>
  );
}
