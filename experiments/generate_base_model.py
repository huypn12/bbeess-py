from data.model_generator.sir_generator import SirGenerator


def generate_base_model(s0: int, i0: int, r0: int):
    generator = SirGenerator(s0, i0, r0)
    generator.run()
    model_file_name = f"sir_{s0}_{i0}_{r0}.pm"
    model_file = f"data/prism/{model_file_name}"
    props_file_name = f"sir_{s0}_{i0}_{r0}.pctl"
    props_file = f"data/prism/{props_file_name}"
    generator.save(model_file, props_file)


if __name__ == "__main__":
    # generate_base_model(3, 1, 0)
    # generate_base_model(5, 1, 0)
    generate_base_model(10, 1, 0)
    # generate_base_model(15, 1, 0)
    # generate_base_model(20, 1, 0)
    # generate_base_model(25, 1, 0)
