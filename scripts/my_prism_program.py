import stormpy
import stormpy.core
import stormpy.pars


class MyPrismProgram(object):
    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
    ) -> None:
        super().__init__()
        self.prism_model_file: str = prism_model_file
        self.prism_props_file: str = prism_props_file
        self.prism_program = stormpy.parse_prism_program(self.prism_model_file)
        props_str = self._load_props_file()
        self.prism_props: str = stormpy.parse_properties_for_prism_program(
            props_str, self.prism_program
        )

    def _load_props_file(self) -> str:
        lines = []
        with open(self.prism_props_file, "r") as fptr:
            lines = fptr.readlines()
        return ";".join(lines)