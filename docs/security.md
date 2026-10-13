# Security model

[中文](zh-CN/security.md)

| Property | Mechanism | Test |
| --- | --- | --- |
| Tool calls only from native tool calling | `decisions.decide` reads assembled native calls; prose never triggers tools | `test_units::test_decisions_come_from_native_calls_never_from_prose` |
| Unknown / forged arguments rejected | strict schemas (`extra="forbid"`) | `test_security::test_the_model_cannot_grant_itself_access_through_arguments` |
| Every call checked | single `ToolExecutor`, policy before handler | `test_executor_policy` |
| Revocation immediate | grants re-read per call | `test_capabilities_from_role_or_grant_and_revocation_is_immediate` |
| No cross-principal access | runs resumable only by their principal; grants bound to principal | `test_only_the_runs_principal_can_resume_it` |
| Approval integrity | id + call id + arguments hash + principal + expiry + single use | `test_approval_validation_refuses_replay_tampering_expiry_and_impostors` |
| Injection containment | scanner flags untrusted output; side-effecting tools locked for the run | `test_injected_tool_output_locks_side_effects_for_the_rest_of_the_run` |
| Untrusted content stays data | tool results enter as `tool` messages; optional notice; never system text | `test_tool_observations_are_marked_as_data_when_a_notice_is_configured` |
| Search privacy | HOSTED never with private context; ISOLATED queries gated | `test_hosted_search_is_never_chosen_when_the_request_is_private` |
| Provider errors don't leak bodies | error messages carry status/code only | `test_http_errors_map_to_stable_codes_without_echoing_the_body` |
| Side effects at most once | idempotency records, reconciliation | `test_a_side_effect_with_unknown_outcome_is_not_retried` |

**Heuristics, not guarantees:** the injection scanner and the query privacy gate. Known limitation (pinned
by `test_known_limitation_short_private_fragments_pass_the_gate`): private fragments shorter than 12
characters pass the gate. The framework never logs tool observations or prompts.
