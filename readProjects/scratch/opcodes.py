def getExistingOpcodesList():
    controll_opcodes = [
        'control_repeat',
        'control_repeat_until',
        'control_while',
        'control_for_each',
        'control_forever',
        'control_wait',
        'control_wait_until',
        'control_if',
        'control_if_else',
        'control_stop',
        'control_create_clone_of',
        'control_delete_this_clone',
        'control_get_counter',
        'control_incr_counter',
        'control_clear_counter',
        'control_all_at_once'
    ]
    data_opcodes = [
        'data_variable',
        'data_setvariableto',
        'data_changevariableby',
        'data_hidevariable',
        'data_showvariable',
        'data_listcontents',
        'data_addtolist',
        'data_deleteoflist',
        'data_deletealloflist',
        'data_insertatlist',
        'data_replaceitemoflist',
        'data_itemoflist',
        'data_itemnumoflist',
        'data_lengthoflist',
        'data_listcontainsitem',
        'data_hidelist',
        'data_showlist'
    ]
    event_opcodes = [
        'event_whentouchingobject',
        'event_broadcast',
        'event_broadcastandwait',
        'event_whenflagclicked',
        'event_whenthisspriteclicked',
        'event_whentouchingobject',
        'event_whenstageclicked',
        'event_whenbackdropswitchesto',
        'event_whengreaterthan',
        'event_whenbroadcastreceived'
    ]
    looks_opcodes = [
        'looks_say',
        'looks_sayforsecs',
        'looks_think',
        'looks_thinkforsecs',
        'looks_show',
        'looks_hide',
        'looks_hideallsprites'  # legacy no-op block
        'looks_switchcostumeto',
        'looks_switchbackdropto',
        'looks_switchbackdroptoandwait',
        'looks_nextcostume',
        'looks_nextbackdrop',
        'looks_changeeffectby',
        'looks_seteffectto',
        'looks_cleargraphiceffects',
        'looks_changesizeby',
        'looks_setsizeto',
        'looks_changestretchby'  # legacy no-op block
        'looks_setstretchto',  # legacy no-op block
        'looks_gotofrontback',
        'looks_goforwardbackwardlayers',
        'looks_size',
        'looks_costumenumbername',
        'looks_backdropnumbername',
    ]
    motion_opcodes = [
        'motion_movesteps',
        'motion_gotoxy',
        'motion_goto',
        'motion_turnright',
        'motion_turnleft',
        'motion_pointindirection',
        'motion_pointtowards',
        'motion_glidesecstoxy',
        'motion_glideto',
        'motion_ifonedgebounce',
        'motion_setrotationstyle',
        'motion_changexby',
        'motion_setx',
        'motion_changeyby',
        'motion_sety',
        'motion_xposition',
        'motion_yposition',
        'motion_direction',
        # Legacy  no - op  blocks:
        'motion_scroll_right',
        'motion_scroll_up',
        'motion_align_scene',
        'motion_xscroll',
        'motion_yscroll'
    ]
    operator_opcodes = [
        'operator_add',
        'operator_subtract',
        'operator_multiply',
        'operator_divide',
        'operator_lt',
        'operator_equals',
        'operator_gt',
        'operator_and',
        'operator_or',
        'operator_not',
        'operator_random',
        'operator_join',
        'operator_letter_of',
        'operator_length',
        'operator_contains',
        'operator_mod',
        'operator_round',
        'operator_mathop',
    ]
    procedures_opcodes = [
        'procedures_definition',
        'procedures_call',
        'argument_reporter_string_number',
        'argument_reporter_boolean'
    ]
    sensing_opcodes = [
        'sensing_touchingobject',
        'sensing_touchingcolor',
        'sensing_coloristouchingcolor',
        'sensing_distanceto',
        'sensing_timer',
        'sensing_resettimer',
        'sensing_of',
        'sensing_mousex',
        'sensing_mousey',
        'sensing_setdragmode',
        'sensing_mousedown',
        'sensing_keypressed',
        'sensing_current',
        'sensing_dayssince2000',
        'sensing_loudness',
        'sensing_loud',
        'sensing_askandwait',
        'sensing_answer',
        'sensing_username',
        'sensing_userid'  # legacy no - op block
    ]
    sound_opcodes = [
        'sound_play',
        'sound_playuntildone',
        'sound_stopallsounds',
        'sound_seteffectto',
        'sound_changeeffectby',
        'sound_cleareffects',
        'sound_sounds_menu',
        'sound_beats_menu',
        'sound_effects_menu',
        'sound_setvolumeto',
        'sound_changevolumeby',
        'sound_volume'
    ]
    all_opcodes = []
    all_opcodes.extend(controll_opcodes)
    all_opcodes.extend(data_opcodes)
    all_opcodes.extend(event_opcodes)
    all_opcodes.extend(looks_opcodes)
    all_opcodes.extend(motion_opcodes)
    all_opcodes.extend(operator_opcodes)
    all_opcodes.extend(procedures_opcodes)
    all_opcodes.extend(sensing_opcodes)
    all_opcodes.extend(sound_opcodes)
    return all_opcodes
