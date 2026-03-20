import re
import torch
from typing import List

from utils.ast_tools import IdentifierAnalyzer, is_valid_identifier, CodeTransformer
from utils.model_zoo import ModelZoo



class CodeBasedCandidateGenerator:
    def __init__(self, model_zoo: ModelZoo, analyzer: IdentifierAnalyzer):
        self.model_zoo = model_zoo
        self.analyzer = analyzer

    def generate_candidates(self, code: str, var_name: str, top_k_mlm=100, top_n_keep=20) -> List[str]:
        # 【修改1】扩大 MLM 的搜索池，因为后面会被 AST 过滤掉很多
        keywords = self.analyzer.keywords
        code_bytes = code.encode("utf-8")
        identifiers = self.analyzer.extract_identifiers(code_bytes)

        if var_name not in identifiers:
            return []

        # 预测掩码
        mask_token = self.model_zoo.mlm_tokenizer.mask_token
        # 注意：这里只 mask 了第一次出现的位置。为了防止它匹配到变量声明(如 int var; 会生成 float 等)，
        # 其实是可以接受的，因为后面的 AST 检查会过滤掉非法的名字。
        masked_code = re.sub(r'\b' + re.escape(var_name) + r'\b', mask_token, code, count=1)

        inputs = self.model_zoo.mlm_tokenizer(
            masked_code, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model_zoo.device)

        mask_token_id = self.model_zoo.mlm_tokenizer.mask_token_id
        mask_token_indices = (inputs.input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_token_indices) == 0:
            return []

        with torch.no_grad():
            logits = self.model_zoo.mlm_model(**inputs).logits
            mask_logits = logits[0, mask_token_indices[0], :]
            # 获取更多候选，弥补清洗造成的损失
            _, top_k_indices = torch.topk(mask_logits, top_k_mlm, dim=-1)

            # 【修改2】彻底清洗 BPE 特殊字符
            raw_candidates = [
                self.model_zoo.mlm_tokenizer.decode([idx]).strip()
                for idx in top_k_indices
            ]

        valid_candidates = []

        # 遍历去重后的候选词
        for raw_cand in set(raw_candidates):
            # 清理 Roberta/CodeBERT 特有的 Ġ 符号和 ## 符号
            cand = raw_cand.replace('Ġ', '').replace('##', '').strip()
            # 移除非字母数字的干扰字符 (有些 token 会包含标点)
            cand = re.sub(r'[^a-zA-Z0-9_]', '', cand)

            # 1. 合法性检查 (确保清洗后依然是合法的变量名)
            if not cand or not cand[0].isalpha() and cand[0] != '_': continue
            # 2. 关键字黑名单
            if cand in keywords: continue
            # 3. 避免重命名为自身
            if cand == var_name: continue

            # 【核心修改点：删除了 if cand in existing_names: continue】
            # 允许变量重名，这能极大干扰模型的注意力机制！

            # 4. AST 作用域检查 (防止破坏语法)
            if not self.analyzer.can_rename_to(code_bytes, var_name, cand):
                continue

            # 5. 代码变换验证 (确保替换不报错)
            try:
                _ = CodeTransformer.validate_and_apply(
                    code_bytes, identifiers, {var_name: cand}, analyzer=self.analyzer
                )
                # 【修改3】直接将合法的词加入，不再计算 embedding similarity (把语义判断交给遗传算法)
                valid_candidates.append(cand)
            except Exception:
                continue

            # 达到数量要求即可提前退出
            if len(valid_candidates) >= top_n_keep:
                break

        return valid_candidates