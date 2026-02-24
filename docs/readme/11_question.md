# 本文档记录阅读时对细节处有疑问的地方，解答时要求尽量详细，引用具体代码，并举一个小例子来回答

## 1, @agent_r1/src/agent_ray_trainer.py中的 def init_workers(self)有疑问。
什么是workers，什么是GPU 资源池，actor, critic, refer等模型都是来自于这里吗？

if self.hybrid_engine: 这里只有这个混合引擎模式意味着必须使用这个模式吗？什么是混合引擎模式

这里的语法是什么意思， ：resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ActorRollout
            )
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool][
                "actor_rollout"
            ] = actor_rollout_cls 



