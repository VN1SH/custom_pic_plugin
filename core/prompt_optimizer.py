"""提示词优化器模块

使用 MaiBot 主 LLM 将用户描述优化为专业的绘画提示词。
纯净调用，不带人设和回复风格。
"""
from typing import Tuple, Optional, Dict, Any
from src.common.logger import get_logger
from src.plugin_system.apis import llm_api

logger = get_logger("prompt_optimizer")

# 提示词优化系统提示词
OPTIMIZER_SYSTEM_PROMPT = """You are a professional AI art prompt engineer. Your task is to convert user descriptions into high-quality English prompts for image generation models (Stable Diffusion, DALL-E, etc.).

## Rules:
1. Output ONLY the English prompt, no explanations or translations
2. Use comma-separated tags/phrases
3. Follow structure: subject, action/pose, scene/background, lighting, style, quality tags
4. Use weight syntax for emphasis: (keyword:1.2) for important elements
5. Keep prompts concise but descriptive (50-150 words ideal)
6. Always end with quality tags: masterpiece, best quality, high resolution

## Examples:

Input: 海边的女孩
Output: 1girl, solo, standing on beach, ocean waves, sunset sky, orange and pink clouds, warm lighting, summer dress, wind blowing hair, peaceful expression, masterpiece, best quality, high resolution

Input: 可爱的猫咪睡觉
Output: cute cat, sleeping, curled up on soft blanket, fluffy fur, closed eyes, peaceful, warm indoor lighting, cozy atmosphere, detailed fur texture, masterpiece, best quality, high resolution

Input: 赛博朋克城市
Output: cyberpunk cityscape, neon lights, futuristic buildings, flying cars, rain, reflective wet streets, holographic advertisements, purple and blue color scheme, atmospheric, cinematic lighting, masterpiece, best quality, high resolution

Now convert the following description to an English prompt:"""

SELFIE_SCENE_SYSTEM_PROMPT = """You are an image prompt planner for "MaiMai selfie".

Task:
- Read the latest chat context and the user's selfie request.
- Infer MaiMai's current situation or activity (e.g., gaming, working, walking outside).
- Output one concise ENGLISH prompt for an image model.

Hard constraints:
1. The photo is MaiMai's own selfie, single female subject.
2. Keep identity consistent with a fixed reference image.
3. The scene should reflect the inferred current situation from context.
4. No explanation. Output prompt only.
5. Avoid NSFW or unsafe content.

Style constraints:
- Use comma-separated tags/phrases.
- Include selfie camera clues (front camera, phone, natural pose).
- Keep length around 25-80 words.
"""


class PromptOptimizer:
    """提示词优化器

    使用 MaiBot 主 LLM 优化用户描述为专业绘画提示词
    """

    def __init__(self, log_prefix: str = "[PromptOptimizer]"):
        self.log_prefix = log_prefix
        self._model_config = None
        self._audit_model_config = None

    def _get_available_models(self) -> Dict[str, Any]:
        """获取主程序可用模型映射。"""
        try:
            models = llm_api.get_available_models()
            return models if isinstance(models, dict) else {}
        except Exception as e:
            logger.error(f"{self.log_prefix} 获取可用模型列表失败: {e}")
            return {}

    def _get_model_config_by_name(
        self,
        model_name: str = "",
        fallback_keys: Tuple[str, ...] = ()
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """按模型槽位名获取配置，失败时按 fallback 顺序回退。"""
        models = self._get_available_models()
        if not models:
            return None, ""

        preferred = (model_name or "").strip()
        if preferred and preferred in models:
            return models[preferred], preferred

        for key in fallback_keys:
            if key in models:
                return models[key], key

        # 最后兜底取第一个，避免功能全失效
        first_key = next(iter(models.keys()))
        return models[first_key], first_key

    def _get_model_config(self):
        """获取可用的 LLM 模型配置"""
        if self._model_config is None:
            try:
                models = llm_api.get_available_models()
                # 使用 replyer 模型（首要回复模型）
                if "replyer" in models:
                    self._model_config = models["replyer"]
                else:
                    logger.warning(f"{self.log_prefix} 没有找到 replyer 模型")
                    return None
            except Exception as e:
                logger.error(f"{self.log_prefix} 获取模型配置失败: {e}")
                return None
        return self._model_config

    def _get_audit_model_config(self):
        """获取自拍审核模型配置，优先使用主程序 planer/planner。"""
        if self._audit_model_config is not None:
            return self._audit_model_config

        model_config, model_key = self._get_model_config_by_name(
            model_name="planner",
            fallback_keys=("planer", "replyer")
        )
        if model_config:
            self._audit_model_config = model_config
            logger.info(f"{self.log_prefix} 自拍审核使用模型: {model_key}")
            return self._audit_model_config
        logger.warning(f"{self.log_prefix} 未找到 planner/planer/replyer 模型")
        return None

    async def _call_with_model(
        self,
        prompt: str,
        request_type: str,
        model_name: str = "",
        fallback_keys: Tuple[str, ...] = ()
    ) -> Tuple[bool, str, str]:
        """按模型槽位调用主程序LLM，返回 (success, response, used_model_key)。"""
        model_config, used_key = self._get_model_config_by_name(model_name, fallback_keys)
        if not model_config:
            return False, "", ""
        try:
            success, response, reasoning, model_name_return = await llm_api.generate_with_model(
                prompt=prompt,
                model_config=model_config,
                request_type=request_type,
            )
            if success and response:
                return True, response, used_key or (model_name_return or "")
            return False, "", used_key
        except Exception as e:
            logger.error(f"{self.log_prefix} LLM 调用失败({used_key}): {e}")
            return False, "", used_key

    async def optimize(self, user_description: str) -> Tuple[bool, str]:
        """优化用户描述为专业绘画提示词

        Args:
            user_description: 用户原始描述（中文或英文）

        Returns:
            Tuple[bool, str]: (是否成功, 优化后的提示词或错误信息)
        """
        if not user_description or not user_description.strip():
            return False, "描述不能为空"

        model_config = self._get_model_config()
        if not model_config:
            # 降级：直接返回原始描述
            logger.warning(f"{self.log_prefix} 无可用模型，降级使用原始描述")
            return True, user_description

        try:
            # 构建完整 prompt
            full_prompt = f"{OPTIMIZER_SYSTEM_PROMPT}\n\nInput: {user_description.strip()}\nOutput:"

            logger.info(f"{self.log_prefix} 开始优化提示词: {user_description[:50]}...")

            # 调用 LLM（不传递 temperature 和 max_tokens，使用模型默认值）
            success, response, reasoning, model_name = await llm_api.generate_with_model(
                prompt=full_prompt,
                model_config=model_config,
                request_type="plugin.prompt_optimize",
            )

            if success and response:
                # 清理响应（移除可能的前缀/后缀）
                optimized = self._clean_response(response)
                logger.info(f"{self.log_prefix} 优化成功 (模型: {model_name}): {optimized[:80]}...")
                return True, optimized
            else:
                logger.warning(f"{self.log_prefix} LLM 返回空响应，降级使用原始描述: {user_description[:50]}...")
                return True, user_description

        except Exception as e:
            logger.error(f"{self.log_prefix} 优化失败: {e}，使用原始描述: {user_description[:50]}...")
            # 降级：返回原始描述
            return True, user_description

    async def generate_selfie_scene_prompt(
        self,
        user_request: str,
        scene_context: str = "",
        model_name: str = ""
    ) -> Tuple[bool, str]:
        """生成基于当前聊天情景的自拍提示词。"""
        request_text = (user_request or "").strip()
        context_text = (scene_context or "").strip()
        if not request_text:
            return False, "描述不能为空"

        try:
            full_prompt = (
                f"{SELFIE_SCENE_SYSTEM_PROMPT}\n\n"
                f"Chat context:\n{context_text or '(empty)'}\n\n"
                f"User request:\n{request_text}\n\n"
                "Output:"
            )

            logger.info(f"{self.log_prefix} 开始生成自拍场景提示词...")
            success, response, used_model_key = await self._call_with_model(
                prompt=full_prompt,
                request_type="plugin.selfie_scene_prompt",
                model_name=model_name,
                fallback_keys=("replyer", "planner", "planer")
            )

            if success and response:
                prompt = self._clean_response(response)
                logger.info(f"{self.log_prefix} 自拍场景提示词生成成功 (模型: {used_model_key}): {prompt[:80]}...")
                return True, prompt

            logger.warning(f"{self.log_prefix} 自拍场景提示词生成失败，回退原始请求")
            return True, request_text

        except Exception as e:
            logger.error(f"{self.log_prefix} 自拍场景提示词生成异常: {e}，回退原始请求")
            return True, request_text

    async def audit_selfie_request(
        self,
        message_text: str,
        scene_context: str = "",
        model_name: str = "",
        allow_replyer_fallback: bool = True
    ) -> Tuple[bool, str]:
        """使用 planer/planner 对自拍请求做二次审核，返回 (是否通过, 原始响应)。"""
        text = (message_text or "").strip()
        context_text = (scene_context or "").strip()
        if not text:
            return False, "EMPTY"

        audit_prompt = (
            "你是群聊内容审核器。你的任务是判断下面这条请求，是否在明确要求机器人“麦麦本人”发自拍/照片。\n\n"
            "判定规则：\n"
            "1. 只有明确向麦麦本人索要自拍/照片，才输出 YES。\n"
            "2. 如果是在要别人照片、泛泛说照片、讨论照片、玩梗，输出 NO。\n"
            "3. 如果对象不明确（看不出是麦麦本人），输出 NO。\n"
            "4. 只输出 YES 或 NO，不要其他内容。\n\n"
            f"最近聊天上下文：\n{context_text or '(empty)'}\n\n"
            f"当前消息：{text}\n\n"
            "只输出：YES 或 NO"
        )

        try:
            fallback_keys = ("planner", "planer", "replyer") if allow_replyer_fallback else ("planner", "planer")
            success, response, used_model_key = await self._call_with_model(
                prompt=audit_prompt,
                request_type="plugin.selfie_request_audit",
                model_name=model_name,
                fallback_keys=fallback_keys
            )
            if not success or not response:
                return False, "NO"
            cleaned = response.strip().upper()
            if "YES" in cleaned and "NO" not in cleaned:
                return True, response
            if cleaned.startswith("YES"):
                return True, response
            return False, response
        except Exception as e:
            logger.error(f"{self.log_prefix} 自拍审核异常: {e}")
            return False, "NO"

    def _clean_response(self, response: str) -> str:
        """清理 LLM 响应

        移除可能的前缀、后缀、引号等
        """
        result = response.strip()

        # 移除可能的 "Output:" 前缀
        prefixes_to_remove = ["Output:", "output:", "Prompt:", "prompt:"]
        for prefix in prefixes_to_remove:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()

        # 移除首尾引号
        if (result.startswith('"') and result.endswith('"')) or \
           (result.startswith("'") and result.endswith("'")):
            result = result[1:-1]

        # 移除多余换行
        result = " ".join(result.split())

        return result


# 全局优化器实例
_optimizer_instance = None

def get_optimizer(log_prefix: str = "[PromptOptimizer]") -> PromptOptimizer:
    """获取提示词优化器实例（单例）"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = PromptOptimizer(log_prefix)
    return _optimizer_instance


async def optimize_prompt(user_description: str, log_prefix: str = "[PromptOptimizer]") -> Tuple[bool, str]:
    """便捷函数：优化提示词

    Args:
        user_description: 用户原始描述
        log_prefix: 日志前缀

    Returns:
        Tuple[bool, str]: (是否成功, 优化后的提示词)
    """
    optimizer = get_optimizer(log_prefix)
    return await optimizer.optimize(user_description)


async def generate_selfie_scene_prompt(
    user_request: str,
    scene_context: str = "",
    model_name: str = "",
    log_prefix: str = "[PromptOptimizer]"
) -> Tuple[bool, str]:
    """便捷函数：生成自拍场景提示词。"""
    optimizer = get_optimizer(log_prefix)
    return await optimizer.generate_selfie_scene_prompt(user_request, scene_context, model_name)


async def audit_selfie_request(
    message_text: str,
    scene_context: str = "",
    model_name: str = "",
    allow_replyer_fallback: bool = True,
    log_prefix: str = "[PromptOptimizer]"
) -> Tuple[bool, str]:
    """便捷函数：用 planer/planner 模型审核自拍请求。"""
    optimizer = get_optimizer(log_prefix)
    return await optimizer.audit_selfie_request(
        message_text,
        scene_context,
        model_name,
        allow_replyer_fallback
    )
