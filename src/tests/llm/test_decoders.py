import unittest
from voxpipe.llm.tools import MalformedToolCall, ToolCall


def tokenwise(text):
    """Split text the way a tokenizer streams it: every symbol and word its own chunk."""
    import re
    return re.findall(r"\w+|\s+|[^\w\s]", text)


class TestTokenSizedChunks(unittest.TestCase):
    """Tags rarely arrive whole: every decoder must cope with them split at any point."""

    def _run(self, decoder, text):
        return list(decoder(iter(tokenwise(text))))

    def _text(self, items):
        return "".join(i for i in items if isinstance(i, str))

    def test_json_call_split_into_tokens(self):
        from voxpipe.llm.decoders import GeneralDecoder, LegacyXMLDecoder
        body = '{"name": "get_time", "arguments": {"zone": "UTC"}}'
        for decoder in (GeneralDecoder(), LegacyXMLDecoder()):
            for open_tag, close_tag in (("<toolcall>", "</toolcall>"), ("<tool_call>", "</tool_call>")):
                with self.subTest(decoder=type(decoder).__name__, tag=open_tag):
                    items = self._run(decoder, f"Sure. {open_tag}{body}{close_tag}")
                    self.assertEqual(self._text(items), "Sure. ")
                    self.assertEqual(items[-1], ToolCall(name="get_time", arguments={"zone": "UTC"}))

    def test_gemma_call_split_into_tokens(self):
        from voxpipe.llm.decoders import GeneralDecoder, GemmaE2BDecoder
        text = 'One moment. <|tool_call>call:note{text:<|"|>buy milk<|"|>,count:2}<tool_call|>'
        for decoder in (GeneralDecoder(), GemmaE2BDecoder()):
            with self.subTest(decoder=type(decoder).__name__):
                items = self._run(decoder, text)
                self.assertEqual(self._text(items), "One moment. ")
                self.assertEqual(items[-1], ToolCall(name="note", arguments={"text": "buy milk", "count": 2}))

    def test_every_split_point_of_a_call(self):
        from voxpipe.llm.decoders import GeneralDecoder
        text = 'Hi <toolcall>{"name": "t", "arguments": {}}</toolcall>'
        for cut in range(1, len(text)):
            with self.subTest(cut=cut):
                items = list(GeneralDecoder()(iter([text[:cut], text[cut:]])))
                self.assertEqual(self._text(items), "Hi ")
                self.assertEqual(items[-1], ToolCall(name="t", arguments={}))

    def test_thoughts_split_into_tokens_are_dropped(self):
        from voxpipe.llm.decoders import GeneralDecoder, GemmaE2BDecoder, LegacyXMLDecoder
        cases = [
            (GeneralDecoder(), "A<|channel>thought\nsecret plan<channel|>B"),
            (GemmaE2BDecoder(), "A<|channel>thought\nsecret plan<channel|>B"),
            (GeneralDecoder(), "A<think>secret plan</think>B"),
            (LegacyXMLDecoder(), "A<think>secret plan</think>B"),
        ]
        for decoder, text in cases:
            with self.subTest(decoder=type(decoder).__name__, text=text):
                self.assertEqual(self._text(self._run(decoder, text)), "AB")

    def test_text_is_not_held_back_longer_than_a_possible_tag(self):
        from voxpipe.llm.decoders import GeneralDecoder
        stream = GeneralDecoder()(iter(["3 < 4 and ", "x<y", "<", "tool", "box> done"]))
        self.assertEqual(list(stream), ["3 < 4 and ", "x<y", "<toolbox> done"])

    def test_angle_bracket_at_end_of_stream_is_flushed(self):
        from voxpipe.llm.decoders import GeneralDecoder
        self.assertEqual("".join(GeneralDecoder()(iter(["a ", "<"]))), "a <")


class TestMalformedCalls(unittest.TestCase):
    def test_unterminated_call_is_reported(self):
        from voxpipe.llm.decoders import GeneralDecoder
        items = list(GeneralDecoder()(iter(["ok ", "<toolcall>", '{"name": "t"'])))
        self.assertEqual(items, ["ok ", MalformedToolCall(raw='{"name": "t"')])

    def test_calls_without_a_usable_name_or_arguments(self):
        from voxpipe.llm.decoders import GeneralDecoder
        bodies = ['{"arguments": {}}', '{"name": "", "arguments": {}}', '{"name": "t", "arguments": [1]}', '"just a string"']
        for body in bodies:
            with self.subTest(body=body):
                items = list(GeneralDecoder()(iter([f"<toolcall>{body}</toolcall>"])))
                self.assertEqual(items, [MalformedToolCall(raw=body)])

    def test_gemma_body_without_call_prefix(self):
        from voxpipe.llm.decoders import GemmaE2BDecoder
        items = list(GemmaE2BDecoder()(iter(["<|tool_call>retrieve{}<tool_call|>"])))
        self.assertEqual(items, [MalformedToolCall(raw="retrieve{}")])


class TestGemmaArguments(unittest.TestCase):
    def _args(self, body):
        from voxpipe.llm.decoders import GemmaE2BDecoder
        (call,) = GemmaE2BDecoder()(iter([f"<|tool_call>call:t{body}<tool_call|>"]))
        return call.arguments

    def test_string_values_may_look_like_keys(self):
        self.assertEqual(self._args('{text:<|"|>buy milk, eggs: 12<|"|>}'), {"text": "buy milk, eggs: 12"})

    def test_string_values_may_contain_quotes_and_newlines(self):
        self.assertEqual(self._args('{text:<|"|>He said "hi"\nthen left<|"|>}'), {"text": 'He said "hi"\nthen left'})

    def test_nested_objects_and_lists(self):
        self.assertEqual(
            self._args('{filter:{tags:[<|"|>a<|"|>,<|"|>b<|"|>],limit:3},strict:true}'),
            {"filter": {"tags": ["a", "b"], "limit": 3}, "strict": True},
        )

    def test_plain_json_arguments(self):
        self.assertEqual(self._args('{"query": "hello", "n": 2}'), {"query": "hello", "n": 2})

    def test_no_arguments(self):
        self.assertEqual(self._args(""), {})


class TestDecoderRegistry(unittest.TestCase):
    def test_get_decoder(self):
        from voxpipe.core.exceptions import ProviderError
        from voxpipe.llm.decoders import DECODERS, StreamDecoder, get_decoder
        for name in ("native", "general", "gemma_e2b", "legacy_xml"):
            self.assertIsInstance(get_decoder(name), StreamDecoder)
            self.assertIsInstance(get_decoder(name), DECODERS[name])
        with self.assertRaises(ProviderError):
            get_decoder("nonsense")


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
        items = list(d(iter(["<|tool_call>call:bad{invalid json!!!}<tool_call|>"])))
        self.assertEqual(items, [MalformedToolCall(raw="call:bad{invalid json!!!}")])

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
        items = list(d(iter(["<toolcall>", big_body, "</toolcall>"])))
        # The decoder gives up on the oversized call instead of buffering without bound.
        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], MalformedToolCall)

    def test_lxd_malformed_json(self):
        from voxpipe.llm.decoders import LegacyXMLDecoder
        d = LegacyXMLDecoder()
        items = list(d(iter(["<toolcall>{bad</toolcall>"])))
        self.assertEqual(items, [MalformedToolCall(raw="{bad")])

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
