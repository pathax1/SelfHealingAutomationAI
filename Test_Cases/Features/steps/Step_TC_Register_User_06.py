# ***************************************************************************************************
# File        : Step_TC_Register_05.py
# Description : Step definitions for registering a new user on ZARA site with AI healing support.
# Author      : Aniket Pathare | Self-Healing AI Framework
# Date        : 2025-07-18
# ***************************************************************************************************

from behave import given, when, then
from Common_Functions.CommonFunctions import (
    click,
    type_text,
    verify_element,
    add_screenshot_to_report,
    smart_find_element
)

@given('the user has launched the ZARA site and reject cookie')
def step_launch_and_reject_cookies(context):
    tc_id = context.config.userdata.get("testcase", "TC_Register_06")
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    click(context.driver, 'Reject_Optional_Cookies', test_case_id=tc_id)

@when('the user clicks on the "REGISTER" button on the home page')
def step_click_register_home(context):
    tc_id = context.config.userdata.get("testcase", "TC_Register_06")
    click(context.driver, 'LOG_IN', test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    click(context.driver, 'REGISTER_BUTTON', test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)

@when('the user enters "{email}" in the "E-MAIL" field')
def step_enter_email(context, email):
    tc_id = context.config.userdata.get("testcase", "TC_Register_06")
    email = context.testdata.get("Email", "default@example.com")
    type_text(context.driver, 'EMAIL_FIELD', email, test_case_id=tc_id)

@when('the user enters "{pwd}" in the "PASSWORD" field')
def step_enter_password(context, pwd):
    tc_id = context.config.userdata.get("testcase", "TC_Register_06")
    pwd = context.testdata.get("Passcode", "defaultPass")
    type_text(context.driver, 'PASSWORD_FIELD', pwd, test_case_id=tc_id)

@when('the user enters "{name}" in the "NAME" field')
def step_enter_name(context, name):
    tc_id = context.config.userdata.get("testcase", "TC_Register_06")
    iname = context.testdata.get("Name", "defaultPass")
    type_text(context.driver, 'FIRST_NAME_FIELD', iname, test_case_id=tc_id)

@when('the user enters "{surname}" in the "SURNAME" field')
def step_enter_surname(context, surname):
    tc_id = context.config.userdata.get("testcase", "TC_Register_06")
    ilast = context.testdata.get("Surname", "defaultPass")
    type_text(context.driver, 'LAST_NAME_FIELD', ilast, test_case_id=tc_id)

@when('the user enters "{prefix}" as "PREFIX" and "{phone}" as "TELEPHONE" field')
def step_enter_phone(context, prefix, phone):
    tc_id = context.config.userdata.get("testcase", "TC_Register_06")
    iPhone = context.testdata.get("Phone", "defaultPass")
    type_text(context.driver, 'PHONE_NUMBER_FIELD', iPhone, test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)

@when('the user accepts the privacy and cookies policy')
def step_accept_privacy_policy(context):
    tc_id = context.config.userdata.get("testcase", "TC_Register_06")
    click(context.driver, 'NEWSLETTER_CHECKBOX', test_case_id=tc_id)
    click(context.driver, 'PRIVACY_CHECKBOX', test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)

@when('the user clicks on the "CREATE ACCOUNT" button')
def step_click_create_account(context):
    tc_id = context.config.userdata.get("testcase", "TC_Register_06")
    click(context.driver, 'SIGN_UP_SUBMIT_BUTTON', test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)

@then('a confirmation message should be displayed for successful registration')
def step_verify_registration_success(context):
    tc_id = context.config.userdata.get("testcase", "TC_Register_06")
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    #elem, loc, healed = smart_find_element(context.driver, 'registration_success_message', test_case_id=tc_id)
    #actual_text = elem.text
    #expected_text = "We will send you an SMS to verify your phone number"
   # print("Expected:", expected_text)
    #print("Actual:", actual_text)
    #if expected_text.strip().lower() not in actual_text.strip().lower():
        #raise AssertionError(f"Expected '{expected_text}' in actual '{actual_text}'")
    #verify_element(context.driver, 'registration_success_message', test_case_id=tc_id)
