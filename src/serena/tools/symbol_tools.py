"""
Language server-related tools for code graph operations
"""

import logging
import os
from collections.abc import Sequence
from copy import copy
from typing import Any

from serena.tools import (
    Tool,
    ToolMarkerDoesNotRequireActiveProject,
    ToolMarkerSymbolicRead,
)
from solidlsp.ls_types import SymbolKind

log = logging.getLogger(__name__)


def _sanitize_symbol_dict(symbol_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize a symbol dictionary inplace by removing unnecessary information.
    """
    # We replace the location entry, which repeats line information already included in body_location
    # and has unnecessary information on column, by just the relative path.
    symbol_dict = copy(symbol_dict)
    s_relative_path = symbol_dict.get("location", {}).get("relative_path")
    if s_relative_path is not None:
        symbol_dict["relative_path"] = s_relative_path
    symbol_dict.pop("location", None)
    # also remove name, name_path should be enough
    symbol_dict.pop("name")
    return symbol_dict


class GetSymbolsOverviewTool(Tool, ToolMarkerSymbolicRead, ToolMarkerDoesNotRequireActiveProject):
    """
    Gets an overview of the top-level symbols defined in a given file.
    """

    def apply(self, relative_path: str, depth: int = 0, max_answer_chars: int = -1, project: str | None = None) -> str:
        """
        Use this tool to get a high-level understanding of the code symbols in a file.
        This should be the first tool to call when you want to understand a new file, unless you already know
        what you are looking for.

        :param relative_path: the relative path to the file to get the overview of
        :param depth: depth up to which descendants of top-level symbols shall be retrieved
            (e.g. 1 retrieves immediate children). Default 0.
        :param max_answer_chars: if the overview is longer than this number of characters,
            no content will be returned. -1 means the default value from the config will be used.
            Don't adjust unless there is really no other way to get the content required for the task.
        :param project: the path or name of the project to use, or None to use the active project
        :return: a JSON object containing info about top-level symbols in the file
        """
        project_instance = self.get_project(project)
        symbol_retriever = self.create_language_server_symbol_retriever(project)
        file_path = os.path.join(project_instance.project_root, relative_path)

        # The symbol overview is capable of working with both files and directories,
        # but we want to ensure that the user provides a file path.
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File or directory {relative_path} does not exist in the project.")
        if os.path.isdir(file_path):
            raise ValueError(f"Expected a file path, but got a directory path: {relative_path}. ")
        result = symbol_retriever.get_symbol_overview(relative_path, depth=depth)[relative_path]
        result_json_str = self._to_json(result)
        return self._limit_length(result_json_str, max_answer_chars)


class FindSymbolTool(Tool, ToolMarkerSymbolicRead, ToolMarkerDoesNotRequireActiveProject):
    """
    Performs a global search using the language server backend.
    """

    # noinspection PyDefaultArgument
    def apply(
        self,
        name_path_pattern: str,
        project: str | None = None,
        depth: int = 0,
        include_body: bool = False,
        include_kinds: list[int] = [],  # noqa: B006
        exclude_kinds: list[int] = [],  # noqa: B006
        substring_matching: bool = False,
        max_answer_chars: int = -1,
    ) -> str:
        """
        Retrieves information on all symbols/code entities (classes, methods, etc.) based on the given name path pattern.
        The returned symbol information can be used for edits or further queries.
        Specify `depth > 0` to also retrieve children/descendants (e.g., methods of a class).

        A name path is a path in the symbol tree *within a source file*.
        For example, the method `my_method` defined in class `MyClass` would have the name path `MyClass/my_method`.
        If a symbol is overloaded (e.g., in Java), a 0-based index is appended (e.g. "MyClass/my_method[0]") to
        uniquely identify it.

        To search for a symbol, you provide a name path pattern that is used to match against name paths.
        It can be
         * a simple name (e.g. "method"), which will match any symbol with that name
         * a relative path like "class/method", which will match any symbol with that name path suffix
         * an absolute name path "/class/method" (absolute name path), which requires an exact match of the full name path within the source file.
        Append an index `[i]` to match a specific overload only, e.g. "MyClass/my_method[1]".

        For call graph and dependency information, use the get_symbol_graph tool instead.

        :param name_path_pattern: the name path matching pattern (see above)
        :param project: the path or name of the project to search in
        :param depth: depth up to which descendants shall be retrieved (e.g. use 1 to also retrieve immediate children;
            for the case where the symbol is a class, this will return its methods).
            Default 0.
        :param include_body: If True, include the symbol's source code. Use judiciously.
        :param include_kinds: Optional. List of LSP symbol kind integers to include. (e.g., 5 for Class, 12 for Function).
            Valid kinds: 1=file, 2=module, 3=namespace, 4=package, 5=class, 6=method, 7=property, 8=field, 9=constructor, 10=enum,
            11=interface, 12=function, 13=variable, 14=constant, 15=string, 16=number, 17=boolean, 18=array, 19=object,
            20=key, 21=null, 22=enum member, 23=struct, 24=event, 25=operator, 26=type parameter.
            If not provided, all kinds are included.
        :param exclude_kinds: Optional. List of LSP symbol kind integers to exclude. Takes precedence over `include_kinds`.
            If not provided, no kinds are excluded.
        :param substring_matching: If True, use substring matching for the last element of the pattern, such that
            "Foo/get" would match "Foo/getValue" and "Foo/getData".
        :param max_answer_chars: Max characters for the JSON result. If exceeded, no content is returned.
            -1 means the default value from the config will be used.
        :return: a list of symbols (with locations) matching the name.
        """
        parsed_include_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in include_kinds] if include_kinds else None
        parsed_exclude_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in exclude_kinds] if exclude_kinds else None
        symbol_retriever = self.create_language_server_symbol_retriever(project)
        symbols = symbol_retriever.find(
            name_path_pattern,
            include_kinds=parsed_include_kinds,
            exclude_kinds=parsed_exclude_kinds,
            substring_matching=substring_matching,
        )
        symbol_dicts = [_sanitize_symbol_dict(s.to_dict(kind=True, location=True, depth=depth, include_body=include_body)) for s in symbols]
        result = self._to_json(symbol_dicts)
        return self._limit_length(result, max_answer_chars)


class FindReferencingSymbolsTool(Tool, ToolMarkerSymbolicRead, ToolMarkerDoesNotRequireActiveProject):
    """
    Finds symbols that reference the given symbol using the language server backend
    """

    # noinspection PyDefaultArgument
    def apply(
        self,
        name_path: str,
        relative_path: str,
        include_kinds: list[int] = [],  # noqa: B006
        exclude_kinds: list[int] = [],  # noqa: B006
        max_answer_chars: int = -1,
        project: str | None = None,
    ) -> str:
        """
        Finds references to the symbol at the given `name_path`. The result will contain metadata about the referencing symbols
        as well as a short code snippet around the reference.

        :param name_path: for finding the symbol to find references for, same logic as in the `find_symbol` tool.
        :param relative_path: the relative path to the file containing the symbol for which to find references.
            Note that here you can't pass a directory but must pass a file.
        :param include_kinds: same as in the `find_symbol` tool.
        :param exclude_kinds: same as in the `find_symbol` tool.
        :param max_answer_chars: same as in the `find_symbol` tool.
        :param project: the path or name of the project to use, or None to use the active project
        :return: a list of JSON objects with the symbols referencing the requested symbol
        """
        project_instance = self.get_project(project)
        include_body = False  # It is probably never a good idea to include the body of the referencing symbols
        parsed_include_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in include_kinds] if include_kinds else None
        parsed_exclude_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in exclude_kinds] if exclude_kinds else None
        symbol_retriever = self.create_language_server_symbol_retriever(project)
        references_in_symbols = symbol_retriever.find_referencing_symbols(
            name_path,
            relative_file_path=relative_path,
            include_body=include_body,
            include_kinds=parsed_include_kinds,
            exclude_kinds=parsed_exclude_kinds,
        )
        reference_dicts = []
        for ref in references_in_symbols:
            ref_dict = ref.symbol.to_dict(kind=True, location=True, depth=0, include_body=include_body)
            ref_dict = _sanitize_symbol_dict(ref_dict)
            if not include_body:
                ref_relative_path = ref.symbol.location.relative_path
                assert ref_relative_path is not None, f"Referencing symbol {ref.symbol.name} has no relative path, this is likely a bug."
                content_around_ref = project_instance.retrieve_content_around_line(
                    relative_file_path=ref_relative_path, line=ref.line, context_lines_before=1, context_lines_after=1
                )
                ref_dict["content_around_reference"] = content_around_ref.to_display_string()
            reference_dicts.append(ref_dict)
        result = self._to_json(reference_dicts)
        return self._limit_length(result, max_answer_chars)


class GetSymbolGraphTool(Tool, ToolMarkerSymbolicRead, ToolMarkerDoesNotRequireActiveProject):
    """
    Gets the full call graph and dependency graph for a symbol.
    Provides complete semantic understanding of how a symbol connects to the rest of the codebase.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        project: str | None = None,
        max_depth: int = 10,
        max_answer_chars: int = -1,
    ) -> str:
        """
        Gets the complete graph of relationships for a symbol, including:
        - calls: what functions/methods this symbol calls (outgoing)
        - called_by: what functions/methods call this symbol (incoming)
        - uses: constants/variables/fields this symbol references
        - used_by: symbols that reference constants/variables defined here
        - extends: parent classes/interfaces (for classes)
        - extended_by: child classes that extend this (for classes)

        Traverses the call graph transitively up to max_depth, but excludes
        dependencies outside the project (node_modules, vendor, etc.).

        Use this tool when you need to understand the full impact of changing a symbol,
        or to understand how a symbol fits into the larger codebase architecture.

        :param name_path: the name path of the symbol (e.g., "MyClass/my_method")
        :param relative_path: the relative path to the file containing the symbol
        :param project: the path or name of the project to use, or None to use the active project
        :param max_depth: maximum depth for transitive traversal (default 10)
        :param max_answer_chars: max characters for the JSON result
        :return: a JSON object with the complete symbol graph
        """
        project_instance = self.get_project(project)
        symbol_retriever = self.create_language_server_symbol_retriever(project)

        # Find the target symbol
        symbols = symbol_retriever.find(name_path, within_relative_path=relative_path)
        if not symbols:
            raise ValueError(f"Symbol '{name_path}' not found in '{relative_path}'")

        symbol = symbols[0]
        rel_path = symbol.location.relative_path
        line = symbol.location.line
        col = symbol.location.column

        if rel_path is None or line is None or col is None:
            raise ValueError(f"Symbol '{name_path}' has incomplete location information")

        lang_server = symbol_retriever.get_language_server(rel_path)
        symbol_kind = symbol.symbol_kind

        # Build the result
        result: dict[str, Any] = {
            "name_path": symbol.get_name_path(),
            "kind": symbol.kind,
            "file": f"{rel_path}:{symbol.get_body_line_numbers()[0]}-{symbol.get_body_line_numbers()[1]}",
        }

        # Helper to check if a path is within the project (not in dependencies)
        def is_project_path(uri_or_path: str) -> bool:
            if uri_or_path.startswith("file://"):
                path = uri_or_path[7:]
                if path.startswith("/") and len(path) > 2 and path[2] == ":":
                    path = path[1:]  # Handle Windows paths like /C:/...
            else:
                path = uri_or_path
            # Check if path is ignored (node_modules, vendor, etc.)
            try:
                return not project_instance.is_ignored_path(path)
            except Exception:
                return True  # If we can't determine, assume it's in project

        # Helper to extract location string from URI and range
        def get_location_str(uri: str, range_info: dict | None = None) -> str:
            """Convert URI and optional range to 'file:start-end' format."""
            path = uri
            if uri.startswith("file://"):
                path = uri[7:]
                if path.startswith("/") and len(path) > 2 and path[2] == ":":
                    path = path[1:]  # Handle Windows paths
            try:
                path = os.path.relpath(path, project_instance.project_root)
            except ValueError:
                pass
            if range_info:
                start_line = range_info.get("start", {}).get("line", 0) + 1  # Convert to 1-based
                end_line = range_info.get("end", {}).get("line", 0) + 1
                return f"{path}:{start_line}-{end_line}"
            return path

        # Helper to format symbol with location
        def format_symbol_with_location(item: dict) -> str | None:
            """Format a symbol as 'name (file:start-end)'."""
            name = item.get("name")
            if not name:
                return None
            uri = item.get("uri", "")
            range_info = item.get("range") or item.get("selectionRange")
            loc = get_location_str(uri, range_info)
            return f"{name} ({loc})"

        # For methods/functions/constructors: get call hierarchy
        if symbol_kind in (SymbolKind.Method, SymbolKind.Function, SymbolKind.Constructor):
            try:
                call_items = lang_server.request_call_hierarchy_prepare(rel_path, line, col)
                if call_items:
                    # Collect all callers and callees with transitive traversal
                    calls: list[str] = []
                    called_by: list[str] = []
                    visited_outgoing: set[str] = set()
                    visited_incoming: set[str] = set()

                    def traverse_outgoing(items: list, depth: int) -> None:
                        if depth > max_depth:
                            return
                        for item in items:
                            item_name = item.get("name")
                            item_uri = item.get("uri", "")
                            if not item_name or item_name in visited_outgoing:
                                continue
                            if not is_project_path(item_uri):
                                continue
                            visited_outgoing.add(item_name)
                            formatted = format_symbol_with_location(item)
                            if formatted:
                                calls.append(formatted)
                            # Get what this function calls (recursive)
                            outgoing = lang_server.request_outgoing_calls(item)
                            if outgoing:
                                next_items = [call.get("to", {}) for call in outgoing]
                                traverse_outgoing(next_items, depth + 1)

                    def traverse_incoming(items: list, depth: int) -> None:
                        if depth > max_depth:
                            return
                        for item in items:
                            incoming = lang_server.request_incoming_calls(item)
                            if incoming:
                                for call in incoming:
                                    caller = call.get("from", {})
                                    caller_name = caller.get("name")
                                    caller_uri = caller.get("uri", "")
                                    if not caller_name or caller_name in visited_incoming:
                                        continue
                                    if not is_project_path(caller_uri):
                                        continue
                                    visited_incoming.add(caller_name)
                                    formatted = format_symbol_with_location(caller)
                                    if formatted:
                                        called_by.append(formatted)
                                    # Traverse up the call chain
                                    traverse_incoming([caller], depth + 1)

                    # Get outgoing calls (what this function calls)
                    for item in call_items:
                        outgoing = lang_server.request_outgoing_calls(item)
                        if outgoing:
                            next_items = [call.get("to", {}) for call in outgoing]
                            traverse_outgoing(next_items, 1)

                    # Get incoming calls (who calls this function)
                    traverse_incoming(call_items, 1)

                    if calls:
                        result["calls"] = calls
                    if called_by:
                        result["called_by"] = called_by

            except Exception as e:
                log.debug(f"Failed to get call hierarchy for {symbol.name}: {e}")

        # For classes/interfaces: get type hierarchy
        if symbol_kind in (SymbolKind.Class, SymbolKind.Interface, SymbolKind.Struct):
            try:
                type_items = lang_server.request_type_hierarchy_prepare(rel_path, line, col)
                if type_items:
                    extends: list[str] = []
                    extended_by: list[str] = []

                    for item in type_items:
                        # Get supertypes (what this class extends)
                        supers = lang_server.request_supertypes(item)
                        if supers:
                            for s in supers:
                                super_name = s.get("name")
                                super_uri = s.get("uri", "")
                                if super_name and is_project_path(super_uri):
                                    formatted = format_symbol_with_location(s)
                                    if formatted:
                                        extends.append(formatted)

                        # Get subtypes (what extends this class) - traverse transitively
                        visited_subs: set[str] = set()

                        def traverse_subtypes(items: list, depth: int) -> None:
                            if depth > max_depth:
                                return
                            for sub_item in items:
                                subs = lang_server.request_subtypes(sub_item)
                                if subs:
                                    for s in subs:
                                        sub_name = s.get("name")
                                        sub_uri = s.get("uri", "")
                                        if not sub_name or sub_name in visited_subs:
                                            continue
                                        if not is_project_path(sub_uri):
                                            continue
                                        visited_subs.add(sub_name)
                                        formatted = format_symbol_with_location(s)
                                        if formatted:
                                            extended_by.append(formatted)
                                        traverse_subtypes([s], depth + 1)

                        traverse_subtypes([item], 1)

                    if extends:
                        result["extends"] = extends
                    if extended_by:
                        result["extended_by"] = extended_by

            except Exception as e:
                log.debug(f"Failed to get type hierarchy for {symbol.name}: {e}")

        # Get constants/variables/fields this symbol actually uses
        # by checking if references to those symbols fall within this symbol's body
        try:
            uses: list[str] = []
            body_start, body_end = symbol.get_body_line_numbers()

            # Collect all constants/variables/fields from:
            # 1. Top-level symbols in the same file
            # 2. Class-level fields if this is a method
            candidate_symbols: list[dict] = []

            # Get file-level symbols
            all_file_symbols = symbol_retriever.find("", within_relative_path=rel_path)
            for file_sym in all_file_symbols:
                sym_kind = file_sym.symbol_kind
                if sym_kind in (SymbolKind.Constant, SymbolKind.Variable, SymbolKind.Field, SymbolKind.Property):
                    # Skip if it's the same symbol we're analyzing
                    if file_sym.name == symbol.name:
                        continue
                    # Skip if it's defined inside the function we're analyzing
                    sym_start, sym_end = file_sym.get_body_line_numbers()
                    if body_start <= sym_start <= body_end:
                        continue
                    candidate_symbols.append({
                        "name": file_sym.name,
                        "line": file_sym.location.line,
                        "col": file_sym.location.column,
                        "start": sym_start,
                        "end": sym_end,
                    })

            # For each candidate, check if it's referenced within our function's body
            for candidate in candidate_symbols:
                try:
                    refs = lang_server.request_references(
                        rel_path,
                        candidate["line"],
                        candidate["col"]
                    )
                    if refs:
                        for ref in refs:
                            ref_range = ref.get("range", {})
                            ref_start_line = ref_range.get("start", {}).get("line", -1)
                            # Check if reference is within our function's body (0-based vs 1-based adjustment)
                            if body_start - 1 <= ref_start_line <= body_end - 1:
                                # Format with location
                                formatted = f"{candidate['name']} ({rel_path}:{candidate['start']}-{candidate['end']})"
                                if formatted not in uses:
                                    uses.append(formatted)
                                break  # Found at least one reference, move to next candidate
                except Exception:
                    pass  # Skip if references lookup fails for this symbol

            if uses:
                result["uses"] = uses

        except Exception as e:
            log.debug(f"Failed to get used symbols for {symbol.name}: {e}")

        # If this symbol is a constant/variable/field, find what functions use it
        if symbol_kind in (SymbolKind.Constant, SymbolKind.Variable, SymbolKind.Field, SymbolKind.Property):
            try:
                used_by: list[str] = []
                used_by_names: set[str] = set()  # Track names to avoid duplicates
                refs = lang_server.request_references(rel_path, line, col)
                if refs:
                    # For each reference, find which function/method contains it
                    for ref in refs:
                        ref_uri = ref.get("uri", "")
                        ref_range = ref.get("range", {})
                        ref_line = ref_range.get("start", {}).get("line", -1)

                        if not is_project_path(ref_uri):
                            continue

                        # Extract relative path from URI
                        ref_rel_path = ref_uri
                        if ref_uri.startswith("file://"):
                            ref_rel_path = ref_uri[7:]
                            if ref_rel_path.startswith("/") and len(ref_rel_path) > 2 and ref_rel_path[2] == ":":
                                ref_rel_path = ref_rel_path[1:]
                        # Make it relative to project root
                        try:
                            ref_rel_path = os.path.relpath(ref_rel_path, project_instance.project_root)
                        except ValueError:
                            continue

                        # Find the function/method that contains this reference
                        try:
                            ref_file_symbols = symbol_retriever.find("", within_relative_path=ref_rel_path)
                            for ref_sym in ref_file_symbols:
                                if ref_sym.symbol_kind in (SymbolKind.Method, SymbolKind.Function, SymbolKind.Constructor):
                                    sym_start, sym_end = ref_sym.get_body_line_numbers()
                                    # Check if reference line is within this function (0-based vs 1-based)
                                    if sym_start - 1 <= ref_line <= sym_end - 1:
                                        if ref_sym.name not in used_by_names and ref_sym.name != symbol.name:
                                            used_by_names.add(ref_sym.name)
                                            # Format with location
                                            formatted = f"{ref_sym.name} ({ref_rel_path}:{sym_start}-{sym_end})"
                                            used_by.append(formatted)
                                        break
                        except Exception:
                            pass

                if used_by:
                    result["used_by"] = used_by

            except Exception as e:
                log.debug(f"Failed to get used_by for {symbol.name}: {e}")

        result_json = self._to_json(result)
        return self._limit_length(result_json, max_answer_chars)
