import unittest
from voxpipe.llm.tools import ToolCall


class TestDecoders(unittest.TestCase):

    def _collect(self, decoder, chunks):
        calls, text = [], []
        for item in decoder(iter(chunks)):
            if isinstance(item, ToolCall):
                calls.append((item.name, item.arguments))
            elif isinstance(item, str):
                text.append(item)
        return calls, "".join(text)

    def _check(self, decoder, chunks, expect_calls, expect_text=""):
        calls, text = self._collect(decoder, chunks)
        self.assertEqual(calls, expect_calls)
        self.assertEqual(text, expect_text)

    # --- GeneralDecoder ---

    def test_gd_plain_text(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, ["Hello! How are you?"], [], "Hello! How are you?")

    def test_gd_gemma_call_format(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "<|tool_call>",
            'call:retrieve{query:<|"|>Elden Ring<|"|>}',
            "<tool_call|>",
        ], [("retrieve", {"query": "Elden Ring"})])

    def test_gd_gemma_with_pretext(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "I'll check! ",
            "<|tool_call>",
            'call:search{query:<|"|>hello<|"|>}',
            "<tool_call|>",
        ], [("search", {"query": "hello"})], "I'll check! ")

    def test_gd_standard_toolcall(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "<toolcall>",
            '{"name": "retrieve", "arguments": {"q": "test"}}',
            "</toolcall>",
        ], [("retrieve", {"q": "test"})])

    def test_gd_html_escaped(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "&lt;toolcall&gt;",
            '{"function": "search", "arguments": {"x": 1}}',
            "&lt;/toolcall&gt;",
        ], [("search", {"x": 1})])

    def test_gd_numeric_args(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "<|tool_call>",
            "call:get{id:42,limit:10}",
            "<tool_call|>",
        ], [("get", {"id": 42, "limit": 10})])

    def test_gd_bool_and_null(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "<|tool_call>",
            "call:check{active:true,data:null}",
            "<tool_call|>",
        ], [("check", {"active": True, "data": None})])

    def test_gd_empty_args(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "<|tool_call>", "call:status{}", "<tool_call|>",
        ], [("status", {})])

    def test_gd_no_tool(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, ["Just a regular response."], [], "Just a regular response.")

    def test_gd_halts_after_first_tool(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "<|tool_call>", 'call:a{x:1}', "<tool_call|>",
            "should not appear",
            "<|tool_call>", 'call:b{y:2}', "<tool_call|>",
        ], [("a", {"x": 1})])

    def test_gd_chunked_streaming(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "pre<|tool_cal",
            "l>call:retrieve{query:<|",
            '"|>hi<|"|>}<tool_call|>post',
        ], [("retrieve", {"query": "hi"})], "pre")

    # --- GemmaE2BDecoder ---

    def test_ge_plain_text(self):
        from voxpipe.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        self._check(d, ["Hello!"], [], "Hello!")

    def test_ge_gemma_call(self):
        from voxpipe.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        self._check(d, [
            "pre ",
            "<|tool_call>",
            'call:retrieve{query:<|"|>hi<|"|>}',
            "<tool_call|>",
            "post",
        ], [("retrieve", {"query": "hi"})], "pre ")

    def test_ge_halts_after_first_tool(self):
        from voxpipe.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        self._check(d, [
            "<|tool_call>", 'call:a{x:1}', "<tool_call|>", "more",
        ], [("a", {"x": 1})])

    # --- LegacyXMLDecoder ---

    def test_lxd_standard(self):
        from voxpipe.llm.decoders import LegacyXMLDecoder
        d = LegacyXMLDecoder()
        self._check(d, [
            "pre ",
            "<toolcall>",
            '{"name": "retrieve", "arguments": {"q": "test"}}',
            "</toolcall>",
        ], [("retrieve", {"q": "test"})], "pre ")

    def test_lxd_function_key(self):
        from voxpipe.llm.decoders import LegacyXMLDecoder
        d = LegacyXMLDecoder()
        self._check(d, [
            "<toolcall>",
            '{"function": "search", "arguments": {"x": 1}}',
            "</toolcall>",
        ], [("search", {"x": 1})])

    # --- Cross-decoder consistency ---

    def test_cross_decoder_gemma_format(self):
        from voxpipe.llm.decoders import GeneralDecoder, GemmaE2BDecoder
        chunks = [
            "<|tool_call>",
            'call:retrieve{query:<|"|>test<|"|>}',
            "<tool_call|>",
        ]
        r1 = list(GeneralDecoder()(iter(chunks)))
        r2 = list(GemmaE2BDecoder()(iter(chunks)))
        self.assertEqual(len(r1), len(r2))
        if r1 and r2:
            self.assertEqual(r1[0].name, r2[0].name)
            self.assertEqual(r1[0].arguments, r2[0].arguments)

    def test_all_decoders_handle_own_format(self):
        """Each decoder must parse its own tag format correctly."""
        from voxpipe.llm.decoders import (
            GeneralDecoder, GemmaE2BDecoder, LegacyXMLDecoder, NativeDecoder
        )

        cases = [
            (GeneralDecoder(), [
                "<|tool_call>",
                'call:retrieve{query:<|"|>test<|"|>}',
                "<tool_call|>",
            ], [("retrieve", {"query": "test"})]),
            (GemmaE2BDecoder(), [
                "<|tool_call>",
                'call:retrieve{query:<|"|>test<|"|>}',
                "<tool_call|>",
            ], [("retrieve", {"query": "test"})]),
            (LegacyXMLDecoder(), [
                "<toolcall>",
                '{"name": "retrieve", "arguments": {"q": "test"}}',
                "</toolcall>",
            ], [("retrieve", {"q": "test"})]),
            (NativeDecoder(), [
                "plain text only",
            ], []),
        ]

        for decoder, chunks, expected in cases:
            with self.subTest(decoder=type(decoder).__name__):
                calls, _ = self._collect(decoder, chunks)
                self.assertEqual(calls, expected)


class TestStreamDecoders(unittest.TestCase):
    """Comprehensive edge-case tests for all decoders."""

    def _collect(self, decoder, chunks):
        calls, text = [], []
        for item in decoder(iter(chunks)):
            if isinstance(item, ToolCall):
                calls.append((item.name, item.arguments))
            elif isinstance(item, str):
                text.append(item)
        return calls, "".join(text)

    def _check(self, decoder, chunks, expect_calls, expect_text=""):
        calls, text = self._collect(decoder, chunks)
        self.assertEqual(calls, expect_calls)
        self.assertEqual(text, expect_text)

    # --- GeneralDecoder edge cases ---

    def test_gd_custom_formats(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder(formats=[
            {"open": "<custom>", "close": "</custom>", "parse": "json"},
        ])
        self._check(d, [
            "<custom>",
            '{"name": "my_tool", "arguments": {}}',
            "</custom>",
        ], [("my_tool", {})])

    def test_gd_empty_formats_list(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder(formats=[])
        self._check(d, ["some plain text"], [], "some plain text")

    def test_gd_thought_tag_dropped(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "before ",
            "<|channel>thought\n",
            "internal monologue here",
            "<channel|>",
            " after",
        ], [], "before  after")

    def test_gd_thought_no_close(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "pre<|channel>thought\n",
            "open thought never closed",
        ], [], "pre")

    def test_gd_tag_in_regular_text(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        # <toolcall> without </toolcall> — opener detected, decoder waits for
        # close that never arrives.  Text after the opener is accumulated as
        # body and dropped when the stream ends.
        self._check(d, ["see <toolcall> in docs"], [], "see ")

    # --- GemmaE2BDecoder edge cases ---

    def test_ge_malformed_json(self):
        from voxpipe.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        calls, text = self._collect(d, [
            "<|tool_call>call:bad{invalid json!!!}<tool_call|>",
        ])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "_parse_error")

    def test_ge_quotes_in_args(self):
        from voxpipe.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        self._check(d, [
            "<|tool_call>",
            'call:search{query:<|"|>it'+chr(39)+'s fine<|"|>}',
            "<tool_call|>",
        ], [("search", {"query": "it's fine"})])

    def test_ge_nested_channel_references(self):
        from voxpipe.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        self._check(d, [
            "text<|channel>thought\n",
            "thinking<channel|>",
            " out",
        ], [], "text out")

    def test_ge_unicode_in_args(self):
        from voxpipe.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        self._check(d, [
            "<|tool_call>",
            'call:find{name:<|"|>caf\\u00e9<|"|>}',
            "<tool_call|>",
        ], [("find", {"name": "caf\u00e9"})])

    def test_ge_empty_stream(self):
        from voxpipe.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        self._check(d, [], [])

    # --- LegacyXMLDecoder edge cases ---

    def test_lxd_buffer_overflow(self):
        from voxpipe.llm.decoders import LegacyXMLDecoder
        d = LegacyXMLDecoder()
        big_body = "{" + "x" * 10_001 + "}"
        calls, text = self._collect(d, [
            "<toolcall>",
            big_body,
            "</toolcall>",
        ])
        # Decoder discards the oversized buffer and yields _parse_error
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "_parse_error")

    def test_lxd_malformed_json(self):
        from voxpipe.llm.decoders import LegacyXMLDecoder
        d = LegacyXMLDecoder()
        calls, text = self._collect(d, [
            "<toolcall>{bad</toolcall>",
        ])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "_parse_error")

    def test_lxd_partial_tag_at_boundary(self):
        from voxpipe.llm.decoders import LegacyXMLDecoder
        d = LegacyXMLDecoder()
        self._check(d, [
            "pre<toolcal",
            'l>{"name":"t","arguments":{}}',
            "</toolcall>",
        ], [("t", {})], "pre")

    def test_lxd_empty_stream(self):
        from voxpipe.llm.decoders import LegacyXMLDecoder
        d = LegacyXMLDecoder()
        self._check(d, [], [])

    # --- NativeDecoder ---

    def test_nd_passthrough(self):
        from voxpipe.llm.decoders import NativeDecoder
        d = NativeDecoder()
        chunks = ["hello", " world"]
        result = list(d(iter(chunks)))
        self.assertEqual(result, ["hello", " world"])

    def test_nd_mixed_content(self):
        from voxpipe.llm.decoders import NativeDecoder
        d = NativeDecoder()
        tc = ToolCall(name="test", arguments={"x": 1})
        chunks = ["a", tc, "b"]
        result = list(d(iter(chunks)))
        self.assertEqual(result, ["a", tc, "b"])

    def test_nd_empty(self):
        from voxpipe.llm.decoders import NativeDecoder
        d = NativeDecoder()
        self.assertEqual(list(d(iter([]))), [])

    # --- StreamDecoder ABC ---

    def test_abc_cannot_instantiate(self):
        from voxpipe.llm.decoders import StreamDecoder
        with self.assertRaises(TypeError):
            StreamDecoder()

    # --- ToolCall passthrough in all decoders ---

    def test_toolcall_passthrough_gd(self):
        from voxpipe.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        tc = ToolCall(name="direct", arguments={"from": "test"})
        result = list(d(iter([tc])))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "direct")
