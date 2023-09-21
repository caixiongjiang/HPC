// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 22-12-1.
#include "parser/parse_expression.hpp"
#include <glog/logging.h>
#include <algorithm>
#include <cctype>
#include <stack>
#include <utility>

namespace kuiper_infer {

void ReversePolish(const std::shared_ptr<TokenNode> &root_node,
                   std::vector<std::shared_ptr<TokenNode>> &reverse_polish) {
  if (root_node != nullptr) {
    ReversePolish(root_node->left, reverse_polish);
    ReversePolish(root_node->right, reverse_polish);
    reverse_polish.push_back(root_node);
  }
}

void ExpressionParser::Tokenizer(bool retokenize) {
  if (!retokenize && !this->tokens_.empty()) {
    return;
  }
  // 判断输入是否为空
  CHECK(!statement_.empty()) << "The input statement is empty!";
  // 移除表达式中的空格
  statement_.erase(std::remove_if(statement_.begin(), statement_.end(),
                                  [](char c) { return std::isspace(c); }),
                   statement_.end());
  CHECK(!statement_.empty()) << "The input statement is empty!";

  for (int32_t i = 0; i < statement_.size();) {
    char c = statement_.at(i);
    if (c == 'a') { // add
      CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'd')
              << "Parse add token failed, illegal character: "
              << statement_.at(i + 1);
      CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'd')
              << "Parse add token failed, illegal character: "
              << statement_.at(i + 2);
      // 初始化一个新的token
      Token token(TokenType::TokenAdd, i, i + 3);
      // 放入token数组中
      tokens_.push_back(token);
      std::string token_operation =
          std::string(statement_.begin() + i, statement_.begin() + i + 3);
      token_strs_.push_back(token_operation);
      i = i + 3;
    } else if (c == 'm') { // mul
      CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'u')
              << "Parse multiply token failed, illegal character: "
              << statement_.at(i + 1);
      CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'l')
              << "Parse multiply token failed, illegal character: "
              << statement_.at(i + 2);
      // 初始化一个新的token
      Token token(TokenType::TokenMul, i, i + 3);
      // 放入token数组中
      tokens_.push_back(token);
      std::string token_operation =
          std::string(statement_.begin() + i, statement_.begin() + i + 3);
      token_strs_.push_back(token_operation);
      i = i + 3;
    } /// 支持sin的词法解析
    else if (c == 's') { // sin
        CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'i')
                << "Parse multiply token failed, illegal character: "
                << statement_.at(i + 1);
        CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'n')
            << "Parse multiply token failed, illegal character: "
            << statement_.at(i + 2);
        // 初始化一个新的token
        Token token(TokenType::TokenSin, i, i + 3);
        tokens_.push_back(token);
        std::string token_operation = std::string(statement_.begin() + i, statement_.begin() + i + 3);
        token_strs_.push_back(token_operation);
        i = i + 3;
    }else if (c == '@') {
      CHECK(i + 1 < statement_.size() && std::isdigit(statement_.at(i + 1)))
              << "Parse number token failed, illegal character: "
              << statement_.at(i + 1);
      // 读取@后面的数字
      int32_t j = i + 1;
      for (; j < statement_.size(); ++j) {
        if (!std::isdigit(statement_.at(j))) {
          break;
        }
      }
      Token token(TokenType::TokenInputNumber, i, j);
      CHECK(token.start_pos < token.end_pos);
      tokens_.push_back(token);
      std::string token_input_number =
          std::string(statement_.begin() + i, statement_.begin() + j);
      token_strs_.push_back(token_input_number);
      i = j;
    } else if (c == ',') {
      // 遇到"，"，直接初始化一个新的token
      Token token(TokenType::TokenComma, i, i + 1);
      tokens_.push_back(token);
      std::string token_comma =
          std::string(statement_.begin() + i, statement_.begin() + i + 1);
      token_strs_.push_back(token_comma);
      i += 1;
    } else if (c == '(') {
      // 遇到"("，直接初始化一个新的token
      Token token(TokenType::TokenLeftBracket, i, i + 1);
      tokens_.push_back(token);
      std::string token_left_bracket =
          std::string(statement_.begin() + i, statement_.begin() + i + 1);
      token_strs_.push_back(token_left_bracket);
      i += 1;
    } else if (c == ')') {
      // 遇到"("，直接初始化一个新的token
      Token token(TokenType::TokenRightBracket, i, i + 1);
      tokens_.push_back(token);
      std::string token_right_bracket =
          std::string(statement_.begin() + i, statement_.begin() + i + 1);
      token_strs_.push_back(token_right_bracket);
      i += 1;
    } else {
      LOG(FATAL) << "Unknown  illegal character: " << c;
    }
  }
}

const std::vector<Token> &ExpressionParser::tokens() const {
  return this->tokens_;
}

const std::vector<std::string> &ExpressionParser::token_strs() const {
  return this->token_strs_;
}

std::shared_ptr<TokenNode> ExpressionParser::Generate_(int32_t &index) {
  CHECK(index < this->tokens_.size());
  const auto current_token = this->tokens_.at(index);
  // current_token代表第一个token，必须要以下面三个类型开头（InputNumber、Add、Mul）"，"、"(" ")"不能为开头
  /// 增加开头的类型检查sin
  CHECK(current_token.token_type == TokenType::TokenInputNumber ||
      current_token.token_type == TokenType::TokenAdd ||
      current_token.token_type == TokenType::TokenMul ||
      current_token.token_type == TokenType::TokenSin);
  // current_token 为 InputNumber的情况
  if (current_token.token_type == TokenType::TokenInputNumber) {
    uint32_t start_pos = current_token.start_pos + 1;
    uint32_t end_pos = current_token.end_pos;
    CHECK(end_pos > start_pos || end_pos <= this->statement_.length())
            << "Current token has a wrong length";
    const std::string &str_number =
        std::string(this->statement_.begin() + start_pos,
                    this->statement_.begin() + end_pos);
    // 初始化一个叶子节点
    return std::make_shared<TokenNode>(std::stoi(str_number), nullptr, nullptr);

  } // current_token 为 mul或者add的情况 需要进行下一层递归构建对应的左子节点和右子节点
  else if (current_token.token_type == TokenType::TokenMul ||
      current_token.token_type == TokenType::TokenAdd) {
    // 创建add表达式的父节点
    std::shared_ptr<TokenNode> current_node = std::make_shared<TokenNode>();
    current_node->num_index = int(current_token.token_type);
    // 指向下一个位置
    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing left bracket!";
    // 判断是否是左括号
    CHECK(this->tokens_.at(index).token_type == TokenType::TokenLeftBracket);

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing correspond left token!";
    const auto left_token = this->tokens_.at(index);


    // 判断当前需要处理的left token是否是合法类型
    // add(mul(@0, @1)，@2) left_token = mul 类型
    // add(@1, @2) left_token = @1
    /// 如果增加了sin语法判断，在遇到add和mul的时候也需要增加语法判断
    if (left_token.token_type == TokenType::TokenInputNumber ||
        left_token.token_type == TokenType::TokenAdd ||
        left_token.token_type == TokenType::TokenMul ||
        left_token.token_type == TokenType::TokenSin) {
      current_node->left = Generate_(index);
    } else {
      LOG(FATAL) << "Unknown token type: " << int(left_token.token_type);
    }

    index += 1;
    // 当前index指向add(@1, @2)中的逗号
    CHECK(index < this->tokens_.size()) << "Missing comma!";
    CHECK(this->tokens_.at(index).token_type == TokenType::TokenComma);

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing correspond right token!";
    const auto right_token = this->tokens_.at(index);
    /// 增加判断条件
    if (right_token.token_type == TokenType::TokenInputNumber ||
        right_token.token_type == TokenType::TokenAdd ||
        right_token.token_type == TokenType::TokenMul ||
        right_token.token_type == TokenType::TokenSin) {
      current_node->right = Generate_(index); /// 进行递归解析
    } else {
      LOG(FATAL) << "Unknown token type: " << int(right_token.token_type);
    }

    index += 1;
    CHECK(index < this->tokens_.size()) << "Missing right bracket!";
    CHECK(this->tokens_.at(index).token_type == TokenType::TokenRightBracket);
    return current_node;
  } /// 增加sin的语法解析代码
  else if (current_token.token_type == TokenType::TokenSin) {
      // 创建sin表达式的父节点
      std::shared_ptr<TokenNode> current_node = std::make_shared<TokenNode>();
      current_node->num_index = int(current_token.token_type);
      index += 1; // 指向下一个位置

      CHECK(index < this->tokens_.size()) << "Missing left bracket!"; // 跳过左括号
      CHECK(this->tokens_.at(index).token_type == TokenType::TokenLeftBracket);
      index += 1; // 指向下一个位置

      const auto cur_token = this->tokens_.at(index);
      // 判断当前需要处理的token是否是合法类型
      // sin(@1) 的token类型为inputNumber
      // sin(mul(@1, @2))的类型为mul
      // sin(add(@1, @2))的类型为add
      // sin(sin(@1))的类型为sin
      if (cur_token.token_type == TokenType::TokenInputNumber ||
            current_token.token_type == TokenType::TokenAdd ||
            current_token.token_type == TokenType::TokenMul ||
            current_token.token_type == TokenType::TokenSin) {
          current_node->left = Generate_(index); // 递归执行generate函数
      } else {
          LOG(FATAL) << "Unknown token type: " << int(cur_token.token_type);
      }

      index += 1;
      CHECK(index < this->statement_.size()) << "Missing right bracket!";
      CHECK(this->tokens_.at(index).token_type == TokenType::TokenRightBracket);
      return current_node;
  }else {
    LOG(FATAL) << "Unknown token type: " << int(current_token.token_type);
  }
}

// 表达式层的操作
std::vector<std::shared_ptr<TokenNode>> ExpressionParser::Generate() {
  if (this->tokens_.empty()) {
    this->Tokenizer(true);
  }
  int index = 0;
  // 调用前面的函数构建语法树
  std::shared_ptr<TokenNode> root = Generate_(index);
  CHECK(root != nullptr);
  CHECK(index == tokens_.size() - 1);

  // 转逆波兰式,之后转移到expression中
  std::vector<std::shared_ptr<TokenNode>> reverse_polish;
  ReversePolish(root, reverse_polish);

  return reverse_polish;
}

// 表达式层的注册
TokenNode::TokenNode(int32_t num_index, std::shared_ptr<TokenNode> left,
                     std::shared_ptr<TokenNode> right)
    : num_index(num_index), left(left), right(right) {}
}  // namespace kuiper_infer