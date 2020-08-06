class Monitor:
    def __init__(self, json):
        # The id
        self.id = json.get('id')
        # The name of the monitors mode: default, large, slider, or list.
        self.mode = json.get('mode')
        # The opcode of the block the monitor belongs to.
        self.opcode = json.get('opcode')
        # An object associating names of inputs of the block the monitor belongs to with their values.
        self.params = json.get('params')
        # The name of the target the monitor belongs to,if any.
        self.spriteName = json.get('spriteName')
        # The value appearing on the monitor.
        self.value = json.get('value')
        # The width.
        self.width = json.get('width')
        # The height.
        self.height = json.get('height')
        # The x - coordinate.
        self.x = json.get('x')
        # The y - coordinate.
        self.y = json.get('y')
        # True if the monitor is visible and false otherwise.
        self.visible = json.get('visible')
        # Monitors that do not belong to lists also have these properties:
        # The minimum value of the monitor's slider.
        self.sliderMin = json.get('sliderMin')
        # The maximum value of the monitor's slider.
        self.sliderMax = json.get('sliderMax')
        # True if the monitor's slider allows only integer values and false otherwise.
        self.isDiscrete = json.get('isDiscrete')
