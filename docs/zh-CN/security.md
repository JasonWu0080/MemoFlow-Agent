# 安全模型

[English](../security.md)

| 属性 | 机制 | 测试 |
| --- | --- | --- |
| 工具调用只来自原生工具调用协议 | `decisions.decide` 读取组装好的原生调用；模型文本永远不会触发工具 | `test_units::test_decisions_come_from_native_calls_never_from_prose` |
| 未知 / 伪造的参数被拒绝 | 严格 schema（`extra="forbid"`） | `test_security::test_the_model_cannot_grant_itself_access_through_arguments` |
| 每次调用都会检查 | 唯一的 `ToolExecutor`，handler 之前先评估策略 | `test_executor_policy` |
| 撤销立即生效 | 每次调用都重新读取授权 | `test_capabilities_from_role_or_grant_and_revocation_is_immediate` |
| 不能跨主体访问 | run 只能由其主体恢复；授权绑定主体 | `test_only_the_runs_principal_can_resume_it` |
| 审批完整性 | id + 调用 id + 参数哈希 + 主体 + 过期 + 只能用一次 | `test_approval_validation_refuses_replay_tampering_expiry_and_impostors` |
| 注入隔离 | 扫描器标记不可信输出；本次 run 中锁定有副作用的工具 | `test_injected_tool_output_locks_side_effects_for_the_rest_of_the_run` |
| 不可信内容始终是数据 | 工具结果以 `tool` 消息进入；可配置数据声明；永不进入 system 文本 | `test_tool_observations_are_marked_as_data_when_a_notice_is_configured` |
| 搜索隐私 | 有私有上下文时绝不 HOSTED；ISOLATED 的查询经过闸门 | `test_hosted_search_is_never_chosen_when_the_request_is_private` |
| Provider 错误不泄露响应体 | 错误消息只含状态码 / 错误码 | `test_http_errors_map_to_stable_codes_without_echoing_the_body` |
| 副作用至多一次 | 幂等记录、对账 | `test_a_side_effect_with_unknown_outcome_is_not_retried` |

**是启发式而非保证：** 注入扫描器和查询隐私闸门。已知限制（由 `test_known_limitation_short_private_fragments_pass_the_gate`
固定）：短于 12 个字符的私有片段能通过闸门。框架从不记录工具 observation 或提示词；但工具 handler 抛出意外异常时会记录 traceback，
因此不要把私有数据放进异常消息。
