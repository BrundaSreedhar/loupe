# Review rules for this codebase

Written by the maintainers, and given to every reviewer alongside the diff. Use it
for the defects this codebase produces repeatedly — the ones a generic reviewer
has no way of knowing about.

Be specific about what is wrong and why. Vague guidance produces vague findings.

## Examples — replace these

- All database access goes through `db/gateway.ts`. A direct `pg.query` call is a
  high-severity finding even when the SQL itself is safe.
- Money is always integer cents. A float touching a currency value is a bug.
- Anything under `handlers/` runs untrusted input. Validation belongs at the top
  of the handler, not in the helpers it calls.
